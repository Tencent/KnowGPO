# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import time
import traceback
from copy import deepcopy
from json import JSONDecodeError
from typing import Any, Optional
from uuid import uuid4
import atexit
from datetime import datetime
import json

import numpy as np
import sglang.srt.entrypoints.engine
import torch
import torch.distributed as dist
from sglang.srt.managers.tokenizer_manager import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    get_ip,
    get_open_port,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from verl import DataProto
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionParsedSchema, OpenAIFunctionToolCall
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.net_utils import is_ipv6
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.config import RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
)
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser

try:
    from sglang.srt.entrypoints.openai.protocol import Tool
except ImportError:
    from sglang.srt.openai_api.protocol import Tool


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# logging tool for sglang multi-turn rollout
class SGLangLogManager:
    def __init__(self):
        self.file_handles = {}
        atexit.register(self.close_all)

    def get_handle(self, log_path):
        if log_path not in self.file_handles:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.file_handles[log_path] = open(log_path, 'a', buffering=1)
        return self.file_handles[log_path]

    def log(self, log_path, event, duration=None, extra=None, workid=None, step=None,**extra_keys):
        handle = self.get_handle(log_path)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
        }
        if duration is not None:
            log_entry["duration_sec"] = duration
        if extra is not None:
            log_entry["extra"] = extra
        if workid is not None:
            log_entry["workid"] = workid
        if step is not None:
            log_entry["step"] = step
        if extra_keys is not None:
            for key in extra_keys:
                log_entry[key] = extra_keys[key]
        ordered_keys = ["timestamp", "event", "duration_sec"] + [k for k in log_entry.keys() if k not in ("timestamp", "event", "duration_sec")]
        ordered_entry = {k: log_entry[k] for k in ordered_keys if k in log_entry}
        handle.write(json.dumps(ordered_entry) + '\n')
        handle.flush()

    def close_all(self):
        for handle in self.file_handles.values():
            handle.close()


# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class
class AsyncEngine(sglang.srt.entrypoints.engine.Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # default to use dummy load format, which need to reload weights in first time
        self._need_reload = True

    async def release_memory_occupation(self, tags: Optional[list[str]] = None):
        """Release GPU occupation temporarily."""
        if tags is None:
            obj = ReleaseMemoryOccupationReqInput()
        else:
            obj = ReleaseMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def resume_memory_occupation(self, tags: Optional[list[str]] = None):
        """Resume GPU occupation."""
        # because __init__ is a sync method, it can not call the async release_memory_occupation
        # have to move release_memory_occupation from __init__ to here
        # For multi-stage awake, we run release weight and kv_cache when we resume weights for the first time.
        if self._need_reload:
            await self.release_memory_occupation()
            self._need_reload = False

        if tags is None:
            obj = ResumeMemoryOccupationReqInput()
        else:
            obj = ResumeMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.resume_memory_occupation(obj, None)

    async def update_weights_from_tensor(self, update_weights_request: UpdateWeightsFromTensorReqInput):
        return await self.tokenizer_manager.update_weights_from_tensor(update_weights_request, None)

    async def flush_cache(self):
        return await self.tokenizer_manager.flush_cache()

    async def abort_request(self, rid: str = "", abort_all: bool = False):
        """Abort a specific request or all requests.

        Args:
            rid: The request ID to abort. If empty and abort_all is False, no action is taken.
            abort_all: If True, abort all running requests regardless of rid.
        """
        return self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)


# NOTE(sgm): add for verl. We can optimize it by making
#  the dataloader yield List[int] without padding.
def _pre_process_inputs(
    pad_token_id,
    prompt_token_ids: torch.Tensor,
) -> torch.Tensor:
    # remove the left padding in the prompt token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    return prompt_token_ids[non_pad_index:]


def _extract_logprob_from_output(output):
    """
    extract log_prob from single sglang inference output
    """

    def _map_each_response(resp):
        input_token_logprobs = resp["meta_info"]["input_token_logprobs"]
        log_probs, output_token_ids = zip(
            *[(log_prob, token_ids) for log_prob, token_ids, _ in input_token_logprobs[1:]], strict=False
        )
        return torch.tensor(output_token_ids), torch.tensor(log_probs)

    output_token_ids, log_probs = _map_each_response(output)
    return output_token_ids, log_probs


# NOTE(linjunrong): adhoc
def _post_process_outputs(processing_class, output):
    try:
        # This is when processing_class is a processor
        tokenizer = processing_class.tokenizer
    except AttributeError:
        try:
            # This is when processing_class is a tokenizer
            tokenizer = processing_class
        except AttributeError as e:
            raise ValueError(f"Cannot get tokenizer from processing_class {processing_class}") from e

    def _map_each_response(resp):
        output_token_logprobs = resp["meta_info"]["output_token_logprobs"]
        log_probs, output_token_ids = zip(
            *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True
        )
        return torch.tensor(output_token_ids), torch.tensor(log_probs)

    out_map = map(lambda x: _map_each_response(x), output)
    batched_output_token_ids = []
    batched_logprobs = []
    for output_token_ids, log_probs in out_map:
        batched_output_token_ids.append(output_token_ids)
        batched_logprobs.append(log_probs)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batched_output_token_ids = pad_sequence(batched_output_token_ids, batch_first=True, padding_value=pad_token_id)
    if len(batched_logprobs) > 0:
        batched_logprobs = pad_sequence(batched_logprobs, batch_first=True, padding_value=pad_token_id)
    return batched_output_token_ids, batched_logprobs


def get_tool_call_parser_type(
    processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
) -> str:
    items = FunctionCallParser.ToolCallParserEnum.items()
    for parser_type, parser_cls in items:
        parser = parser_cls()
        try:
            # This is when processing_class is a tokenizer
            tokenizer_vocab = processing_class.get_vocab()
        except AttributeError:
            try:
                # This is when processing_class is a processor
                tokenizer_vocab = processing_class.tokenizer.get_vocab()
            except AttributeError as e:
                raise ValueError(f"Cannot get vocab from processing_class {processing_class}") from e

        if parser.bot_token.strip() in tokenizer_vocab and (
            parser.eot_token == "" or parser.eot_token.strip() in tokenizer_vocab
        ):
            return parser_type
    else:
        raise ValueError(f"No tool call parser found for processing_class {processing_class}")


class SGLangRollout(BaseRollout):
    def __init__(
        self,
        actor_module: str,
        config: RolloutConfig,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        """Synchronized SGLang rollout engine.

        Args:
            actor_module: Huggingface model name or path to the model. The
                model should be supported by SGLang.
            config: A DictConfig object containing SGLang-specific operational
                parameters and rollout settings.
                Refer to https://docs.sglang.ai/backend/server_arguments.html
            processing_class: The tokenizer or processor instance compatible with the actor_module.
            model_hf_config: The Hugging Face model's configuration (e.g.,
                `transformers.PretrainedConfig`). It provides architectural
                details and hyperparameters like `max_position_embeddings`,
                used by SGLang for correct model initialization. This is
                the model's inherent design, not SGLang's runtime behavior.
            port: Optional port for multi-node initialization when nnodes > 1.
            trust_remote_code: Whether or not to allow for custom models
                defined on the Hub in their own modeling files.
            device_mesh: Optional `DeviceMesh` object for distributed setup.
            **kwargs: Additional keyword arguments, primarily `train_tp` for
                Megatron Backend integration to initialize hybrid engine
                process groups.
        """
        super().__init__()
        # self.step = 0
        self.config = config
        self._device_mesh_cpu = device_mesh
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        # 添加的 log_manager
        self.log_manager = SGLangLogManager()
        # 添加的 log_dir，如果您的实验没有设置 EXPERIMENT_NAME，则默认使用 multiturn_log_dir
        logdir = self.config.get("engine_kwargs", {}).get("logdir", "logs")
        self.log_dir = f"{logdir}/multiturn_log_dir"
        self._rank = None  # wsill be set in _init_distributed_env

        (
            self._tool_schemas,
            self._tool_map,
            self._tool_call_parser_type,
            self._sgl_tools,
            self._function_call_parser,
        ) = self._initialize_tools(config, processing_class)
        self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(config)
        # If turn on `free_cache_engine`, SGLang engine's KV cache
        # will be freed after each `generate_sequences` call.
        logger.info(
            f"tool_schemas: {self._tool_schemas}, tool_map: {self._tool_map}, tool_call_parser_type: "
            f"{self._tool_call_parser_type}, sgl_tools: {self._sgl_tools}, function_call_parser: "
            f"{self._function_call_parser}"
        )
        print(f"tool_schemas: {self._tool_schemas}, tool_map: {self._tool_map}, tool_call_parser_type: "
            f"{self._tool_call_parser_type}, sgl_tools: {self._sgl_tools}, function_call_parser: "
            f"{self._function_call_parser}")

        self._init_distributed_env(device_mesh_cpu=device_mesh, **kwargs)

        self._verify_config(model_hf_config=model_hf_config)
        # initialize the inference engine
        self._init_inference_engine(trust_remote_code, actor_module, port)

        self._init_sampling_params(**kwargs)

        self.processing_class = processing_class

        print(f"[DEBUG] chat_template: {self.processing_class.chat_template}")

        try:
            # This is when processing_class is a tokenizer
            self.pad_token_id = self.processing_class.pad_token_id
        except AttributeError:
            try:
                # This is when processing_class is a processor
                self.pad_token_id = self.processing_class.tokenizer.pad_token_id
            except AttributeError as e:
                raise ValueError(f"Cannot get pad_token_id from processing_class {self.processing_class}") from e

    def _init_distributed_env(self, device_mesh_cpu, **kwargs):
        self._device_mesh_cpu = device_mesh_cpu
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert self.tensor_parallel_size <= dist.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        self.train_tp = kwargs.get("train_tp", None)
        if self.train_tp is not None:
            # deployed with megatron
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // self.tensor_parallel_size
            sglang_ps.initialize_parallel_state(
                tensor_model_parallel_size=self.tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        tp_size = self.tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        if self._device_mesh_cpu is None:
            device_mesh_kwargs = dict(
                mesh_shape=(world_size // tp_size, tp_size, 1),
                mesh_dim_names=["dp", "tp", "pp"],
            )

            self._device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

        self._rank = self._device_mesh_cpu.get_rank()
        self._tp_rank = self._device_mesh_cpu["tp"].get_local_rank()
        self._tp_size = self._device_mesh_cpu["tp"].size()
        if self._rank == 0:
            logger.info(f"_init_distributed_env: :tp_world: {self._tp_size}, global_world: {world_size}")
        # get tp_rank of this process in this tp group
        visible_devices = [None] * self._device_mesh_cpu.size(1)

        torch.distributed.all_gather_object(
            visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], self._device_mesh_cpu.get_group("tp")
        )
        self.visible_devices_set = set(",".join(visible_devices).split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))

    def _verify_config(self, model_hf_config):
        if not self.config.get("max_model_len", None):
            self.config.max_model_len = self.config.prompt_length + self.config.response_length
        assert (
            self.config.max_model_len >= self.config.prompt_length + self.config.response_length
        ), f"""max_model_len should be greater than total sequence length (prompt_length + response_length):
            {self.config.max_model_len} >= {self.config.prompt_length} + {self.config.response_length}"""
        max_position_embeddings = None
        if hasattr(model_hf_config, "max_position_embeddings"):
            max_position_embeddings = model_hf_config.max_position_embeddings
        elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
            max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
        elif hasattr(model_hf_config, "text_config") and hasattr(
            model_hf_config.text_config, "max_position_embeddings"
        ):
            max_position_embeddings = model_hf_config.text_config.max_position_embeddings
        if max_position_embeddings is None:
            raise ValueError("max_position_embeddings not found in model_hf_config")
        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            assert max_position_embeddings >= self.config.prompt_length + self.config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= self.config.prompt_length + self.config.response_length
            ), (
                f"model context length should be greater than total sequence length, "
                f"got rope_scaling_factor={rope_scaling_factor} and "
                f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        # currently max_assistant_turns stand for max number of tool calls
        if self.config.multi_turn.max_assistant_turns is None:
            self.config.multi_turn.max_assistant_turns = self.config.max_model_len // 3
        if self.config.multi_turn.max_user_turns is None:
            self.config.multi_turn.max_user_turns = self.config.max_model_len // 3

        print(f"[DEBUG] max_assistant_turns: {self.config.multi_turn.max_assistant_turns}")
        print(f"[DEBUG] max_user_turns: {self.config.multi_turn.max_user_turns}")

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        # initialize the inference engine
        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group("tp"),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0
        engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}

        # attention backend will be changed to fa3 if not specified
        attention_backend = engine_kwargs.pop("attention_backend", None)

        if first_rank_in_node:
            rank = dist.get_rank()
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = AsyncEngine(
                model_path=actor_module,
                dtype=self.config.dtype,
                mem_fraction_static=self.config.gpu_memory_utilization,
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                trust_remote_code=trust_remote_code,
                # NOTE(linjunrong): add rank to prevent SGLang generate same port inside PortArgs.init_new
                # when random.seed is being set during training
                port=30000 + rank,
                # NOTE(Chenyang): turn on log_level to see the decoding speed of SGLang Engine
                # log_level="INFO",
                # NOTE(Chenyang): turn the following lines to see the input and output of each request
                # log_requests=True,
                # log_requests_level=2,
                # NOTE(Chenyang): turn on max_running_requests to set the max concurrent running requests
                # max_running_requests=self.config.max_running_requests,
                mm_attention_backend="fa3",
                attention_backend=attention_backend if attention_backend is not None else "fa3",
                # In async mode for AgentLoop, SGLang support token in token out to avoid the tokenizer
                # inconsistency issue.
                skip_tokenizer_init=self.config.mode == "async",
                **engine_kwargs,
            )
        else:
            self._engine = None

        self.sharding_manager = None
        self.is_sleep = True

    def _init_sampling_params(self, **kwargs):
        kwargs = dict(
            n=1,
            max_new_tokens=self.config.response_length,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.05,
        )
        # supporting adding any sampling params from the config file
        for k in self.config.keys():
            if hasattr(SamplingParams(), str(k)) or "stop" in str(k):
                kwargs[k] = self.config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        self.sampling_params = kwargs

    def _initialize_tools(self, config, processing_class):
        """Initialize tools from configuration.

        Args:
            config: Configuration object containing tool-related settings,
                    specifically `config.multi_turn.tool_config_path`.
            tokenizer: The tokenizer instance used for parsing tool calls from
                       the model's generated text.

        Returns:
            tuple: A tuple containing:
                - tool_schemas (list[dict]): OpenAI-formatted JSON schemas
                  defining each tool's capabilities.
                - tool_map (dict[str, BaseTool]): A dictionary mapping tool
                  names to their executable `BaseTool` objects.
                - tool_call_parser_type (str): The identifier for the specific
                  parser type (e.g., 'json_mode', 'tool_code') used to extract
                  tool calls.
                - sgl_tools (list[sglang.srt.openai_api.protocol.Tool]): Tool
                  definitions optimized for SGLang's internal engine.
                - function_call_parser (sglang.srt.function_call_parser.FunctionCallParser):
                  The active parser instance responsible for extracting
                  structured tool calls from model outputs.
        """
        if config.multi_turn.tool_config_path is None:
            return [], {}, None, [], None

        tools_config_file = config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tools_config_file)

        logger.info(f"Initialize tools from configuration.: tool_list: {tool_list}")
        print(f"Initialize tools from configuration.: tool_list: {tool_list}")
        tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
        tool_map = {tool.name: tool for tool in tool_list}
        tool_call_parser_type = get_tool_call_parser_type(processing_class)
        sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
        function_call_parser = FunctionCallParser(
            sgl_tools,
            tool_call_parser_type,
        )

        return (
            tool_schemas,
            tool_map,
            tool_call_parser_type,
            sgl_tools,
            function_call_parser,
        )

    def _initialize_interactions(self, config):
        """Initialize interactions from configuration.

        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if config.multi_turn.interaction_config_path is None:
            return {}

        interaction_config_file = config.multi_turn.interaction_config_path
        interaction_map = initialize_interactions_from_config(interaction_config_file)

        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        # 从 meta_info 获取 step，但使用 get() 避免 KeyError
        step = prompts.meta_info.get("global_steps", 0)  # 默认值为 0
        self.step = step  # 保留原有逻辑
        
        if self.config.multi_turn.enable:
            try:
                return self._req_level_generate_sequences(prompts, step=step, **kwargs)
            except Exception as e:
                error_msg = f"[rank={self._rank}, tp_rank={self._tp_rank}, step={step}] 调用_req_level_generate_sequences异常: {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                print(error_msg)
                import traceback
                tb = traceback.format_exc()
                logger.error(tb)
                print(tb)
                raise
        try:
            return self._batch_level_generate_sequences(prompts, **kwargs)
        except Exception as e:
            error_msg = f"[rank={self._rank}, tp_rank={self._tp_rank}, step={step}] 调用_batch_level_generate_sequences异常: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
            import traceback
            tb = traceback.format_exc()
            logger.error(tb)
            print(tb)
            raise

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _batch_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generates single-turn sequences for a batch of prompts.
        For single-turn generation, all prompts are processed in one request.
        `_batch_level_generate_sequences` involves:
        1.  Extracting and pre-processing prompt token IDs from the input
            `prompts`. This includes handling padding and preparing raw
            token ID lists.
        2.  Preparing inputs for the SGLang engine, including multi-modal
            data if present.
        3.  Invoking the SGLang engine (`self._engine.async_generate`,
            an async coroutine) with the batch of processed inputs and
            specified sampling parameters on the master TP rank.
        4.  Broadcasting the results from the master TP rank to all
            other TP ranks.
        5.  Post-processing the engine's output to format the generated
            token IDs and (if applicable) log probabilities.
        6.  Constructing the final sequences by concatenating original
            prompts with the generated responses.
        7.  Updating attention masks and position IDs to reflect the full
            concatenated sequences.
        8.  If `self.config.free_cache_engine` is true, the SGLang engine's
            KV cache is flushed after generation on the master TP rank.
        Args:
            prompts: A `DataProto` object containing the batch of
              input prompts, including tensor data (like `input_ids`,
              `attention_mask`) and meta-information (like `eos_token_id`,
              `do_sample`).
            **kwargs: Additional keyword arguments that can override the
              default sampling parameters (e.g., `temperature`, `top_p`,
              `max_new_tokens`). These are temporarily applied using
              `update_sampling_params`.
        Returns:
            DataProto: A `DataProto` object containing the batch of
              generated sequences. This includes tensors for `prompts`
              (original input IDs), `responses` (generated token IDs),
              `input_ids` (concatenated prompt and response),
              `attention_mask`, and `position_ids` for the full
              sequences.
        Note that in GRPO, if the prompts are validated, we repeat the prompts for rollout.n times in ray_trainer.
        Thus we do not need to repeat the prompts here and set the sampling parameter n to 1.
        """
        # input ids: (bs, prompt_length), left-padded
        idx = prompts.batch["input_ids"]
        # attention_mask: (bs, seq_length), left-padded
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to generate attention mask for the
        # response based on EOS token position
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        # Extract non-tensor data
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]).tolist() for i in range(batch_size)],
                dtype=object,
            )

        if "multi_modal_data" in non_tensor_batch:
            sglang_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
                strict=True,
            ):
                sglang_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                        "image_data": (
                            multi_modal_data.get("image", None) if isinstance(multi_modal_data, dict) else None
                        ),
                    }
                )
        else:
            sglang_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in sglang_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        # Extract token IDs and image data for SGLang Engine
        idx_list = [input_data["prompt_token_ids"] for input_data in sglang_inputs]
        image_list = [input_data.get("image_data", None) for input_data in sglang_inputs]

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        # Create request-level sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update(
                {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.05,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "ignore_eos": False,
                    "min_new_tokens": 0,
                    "max_new_tokens": self.config.response_length,
                    "skip_special_tokens": True,
                    "spaces_between_special_tokens": True,
                }
            )
        elif is_validate:
            request_sampling_params.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            )

        # Update with any additional kwargs
        request_sampling_params.update(kwargs)
        torch.cuda.synchronize()
        engine_call_start_time = time.time()
        if self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            output = loop.run_until_complete(
                self._engine.async_generate(
                    prompt=None,  # because we have already convert it to prompt token id
                    sampling_params=request_sampling_params,
                    return_logprob=True,
                    input_ids=idx_list,
                    image_data=image_list,
                )
            )
            torch.cuda.synchronize()
            generate_end_time = time.time()
            log_path = os.path.join(
                self.log_dir,
                f"step_{self.step}",
                f"worker_{self._rank}.jsonl"
            )
            self.log_manager.log(
                log_path,
                event="engine_async_generate",
                duration=generate_end_time - engine_call_start_time,
                workid=self._rank,
                step=self.step
            )
        else:
            output = None

        torch.cuda.synchronize()
        barrier_start_time = time.time()
        # Most naive implementation, can extract tensor and send via gloo if too slow
        dist.barrier()
        torch.cuda.synchronize()
        barrier_end_time = time.time()
        if self._tp_rank == 0:
            log_path = os.path.join(
                self.log_dir,
                f"step_{self.step}",
                f"worker_{self._rank}.jsonl"
            )
            self.log_manager.log(
                log_path,
                event="barrier",
                duration=barrier_end_time - barrier_start_time,
                workid=self._rank,
                step=self.step
            )
        [output] = broadcast_pyobj(
            data=[output],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )
        out = _post_process_outputs(self.processing_class, output)

        response = out[0].to(idx.device)
        rollout_log_probs = None
        if self.config.calculate_log_probs:
            rollout_log_probs = out[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_sequence_to_length(
                    rollout_log_probs, self.config.response_length, self.pad_token_id
                )

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        # free cache engine
        if self._engine is not None and self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def _async_rollout_a_request(
        self,
        req: AsyncRolloutRequest,
        do_sample: bool = True,
        is_validate: bool = False,
        **kwargs,
    ) -> AsyncRolloutRequest:
        """异步执行单个请求的 rollout 过程。
        
        关键变量说明：
        - req: AsyncRolloutRequest - 原始请求对象，包含初始的 prompt_ids、prompt_mask 等
        - _req: AsyncRolloutRequest - req 的深拷贝，在此函数中被修改（添加消息、更新状态等）
        - finish_reason_type: FinishReasonTypeEnum - 结束原因（STOP/LENGTH/TOOL_CALL）
        - output: dict - 引擎单次生成的输出，包含 text、output_ids、meta_info 等字段
        
        计数器变量：
        - current_turns: int - 当前已完成的 assistant turn 数量（每次 LLM 生成算一次）
        - current_docs: int - 累计检索到的文档数量（tool_call 中统计）
        - user_turns: int - 用户交互轮次（interaction 相关）
        - user_turn_rewards: list[float] - 每个 user turn 的奖励列表
        
        消息和工具调用：
        - _req.messages: list[Message] - 完整对话历史，每个 Message 包含 role、content、tool_calls
        - _req.messages[-1].tool_calls: list[OpenAIFunctionToolCall] - 最后一条消息的工具调用列表
        
        === Token 序列三者关系（Multi-turn Tool Use 场景）===
        
        以下是在 multi-turn tool use 场景中，三个变量的演变过程：
        
        消息序列示例：
        [sys, user, assistant(think+tool_call), tool(response), assistant(think+tool_call), 
         tool(response), assistant(think+answer)]
        
        _req.prompt_ids: torch.Tensor [1, prompt_len] 
        ├─ 定义：初始对话的 token IDs（系统提示 + 用户问题）
        ├─ 初始化：在 AsyncRolloutRequest 初始化时设置（initialize_request）
        ├─ 内容：tokenize([sys, user]) + generation_prompt（如 "<im_start>assistant\n"）
        └─ 变化：**在整个 rollout 过程中保持不变**，始终是原始的 prompt
        
        _req.input_ids: torch.Tensor [1, current_seq_len]
        ├─ 定义：当前完整的 token 序列（prompt + 所有已生成的内容）
        ├─ 初始化：初始等于 prompt_ids
        ├─ 更新时机：每次调用以下方法时都会增长
        │   ├─ add_assistant_message(): 添加 assistant 生成的 think + tool_call
        │   ├─ add_tool_response_messages(): 添加 tool 返回的观察结果
        │   └─ add_user_message(): 添加用户交互消息（interaction 场景）
        ├─ 更新方式：_update_input_ids() -> torch.cat([input_ids, new_content_ids], dim=-1)
        └─ 内容演变示例：
            初始: [sys, user, <im_start>assistant\n]
            Turn1: [sys, user, assistant(think), assistant(tool_call), tool(obs1)]
            Turn2: [sys, user, assistant(think), assistant(tool_call), tool(obs1), 
                   assistant(think), assistant(tool_call), tool(obs2)]
            TurnN: [sys, user, ..., assistant(final_answer)]
        
        _req.response_ids: torch.Tensor [1, response_len]
        ├─ 定义：仅包含生成部分的 token IDs（不包括原始 prompt）
        ├─ 初始化：None（在 rollout 开始时）
        ├─ 计算：在 finalize() 方法中计算
        │   └─ response_ids = input_ids[:, prompt_ids.shape[-1]:]
        └─ 内容：从 prompt 结束位置到 input_ids 结束的所有 token
            例如: [assistant(think), assistant(tool_call), tool(obs1), 
                   assistant(think), assistant(tool_call), tool(obs2), 
                   assistant(final_answer)]
        
        关键点：
        1. prompt_ids 固定不变，是训练时的"输入"部分
        2. input_ids 动态增长，是完整的对话序列
        3. response_ids = input_ids - prompt_ids，是训练时的"输出"部分
        4. 在 multi-turn 中，response_ids 包含多轮 assistant 生成和 tool 观察的交替内容
        
        其他关键字段：
        - _req.request_id: str - 请求的唯一标识符
        - _req.state: AsyncRolloutRequestStateEnum - 请求状态（PENDING/RUNNING/TOOL_CALLING/INTERACTING/COMPLETED）
        """
        assert self._tp_rank == 0, "only the master process can call this function"
        _req = deepcopy(req)
        finish_reason_type = None  # 结束原因类型（STOP/LENGTH/TOOL_CALL）
        output = None  # 引擎输出，dict: {"text": str, "output_ids": list[int], "meta_info": {...}}

        # === 计数器变量 ===
        current_turns = 0  # int: 当前 assistant turn 计数，每次 LLM 生成后 +1
        current_docs = 0  # int: 累计检索文档数，从 tool_metrics["total_results"] 累加
        user_turns = 0  # int: 用户交互轮次（interaction 场景）
        user_turn_rewards = []  # list[float]: 每个 user turn 的奖励

        # ===================================
        # Multi-turn Tool Use 完整流程示例
        # ===================================

        # 初始状态（initialize_request 后）:
        # ├─ messages: [{"role": "system", ...}, {"role": "user", "content": "query"}]
        # ├─ prompt_ids: [1, 50] = tokenize([sys, user, <im_start>assistant\n])
        # ├─ input_ids: [1, 50] = prompt_ids
        # └─ response_ids: None

        # Turn 1 - RUNNING -> TOOL_CALLING:
        # ├─ Engine 生成 -> output["text"] = "<think>分析问题</think>\n<tool_call>search(...)</tool_call>"
        # ├─ add_assistant_message() 后:
        # │   ├─ messages: [..., {"role": "assistant", "content": "...", "tool_calls": [...]}]
        # │   ├─ prompt_ids: [1, 50] (不变)
        # │   ├─ input_ids: [1, 200] (新增 assistant token)
        # │   └─ response_ids: None
        # ├─ Tool 执行 -> tool_call_results = [("观察结果1", 0.0, {"total_results": 3})]
        # ├─ add_tool_response_messages() 后:
        # │   ├─ messages: [..., {"role": "tool", "content": "观察结果1"}]
        # │   ├─ prompt_ids: [1, 50] (不变)
        # │   ├─ input_ids: [1, 350] (新增 tool response token)
        # │   └─ response_ids: None
        # └─ current_docs: 3

        # Turn 2 - RUNNING -> TOOL_CALLING:
        # ├─ Engine 生成 -> output["text"] = "<think>继续分析</think>\n<tool_call>search(...)</tool_call>"
        # ├─ add_assistant_message() 后:
        # │   ├─ messages: [..., {"role": "assistant", "content": "...", "tool_calls": [...]}]
        # │   ├─ prompt_ids: [1, 50] (不变)
        # │   ├─ input_ids: [1, 500] (新增 assistant token)
        # │   └─ response_ids: None
        # ├─ Tool 执行 -> tool_call_results = [("观察结果2", 0.0, {"total_results": 2})]
        # ├─ add_tool_response_messages() 后:
        # │   ├─ messages: [..., {"role": "tool", "content": "观察结果2"}]
        # │   ├─ prompt_ids: [1, 50] (不变)
        # │   ├─ input_ids: [1, 650] (新增 tool response token)
        # │   └─ response_ids: None
        # └─ current_docs: 5

        # Turn 3 - RUNNING -> STOP (最终答案):
        # ├─ Engine 生成 -> output["text"] = "<think>综合信息</think>\n最终答案是..."
        # ├─ add_assistant_message() 后:
        # │   ├─ messages: [..., {"role": "assistant", "content": "最终答案是..."}]
        # │   ├─ prompt_ids: [1, 50] (不变)
        # │   ├─ input_ids: [1, 800] (新增 assistant final answer token)
        # │   └─ response_ids: None
        # └─ current_turns: 3, finish_reason_type: STOP

        # Finalize 后:
        # ├─ messages: [sys, user, assistant1, tool1, assistant2, tool2, assistant3]
        # ├─ prompt_ids: [1, 50] (始终不变)
        # ├─ input_ids: [1, 800] (完整序列)
        # ├─ response_ids: [1, 750] = input_ids[:, 50:] (所有生成内容)
        # ├─ response_loss_mask: [1, 750]
        # │   └─ [1,1,...,1, 0,0,...,0, 1,1,...,1, 0,0,...,0, 1,1,...,1]
        # │       ^assistant1  ^tool1     ^assistant2  ^tool2     ^assistant3
        # └─ reward_scores: {"search_tool": [0.5, 0.3], ...}
        # ==================================


        # Turn-level timing tracking
        turn_timings = []

        # request级别完整时间记录
        torch.cuda.synchronize()
        request_start_time = time.time()
        log_path = os.path.join(
            self.log_dir,
            f"step_{self.step}",
            f"worker_{self._rank}.jsonl"
        )

        self.log_manager.log(
            log_path,
            event="request_start",
            extra={
                "request_id": req.request_id,
                "initial_prompt_length": len(req.get_generation_prompt_ids(self.processing_class))
            },
            workid=self._rank,
            step=self.step
        )
        # Create request-level sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update(
                {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.05,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "ignore_eos": False,
                    "min_new_tokens": 0,
                    "max_new_tokens": self.config.response_length,
                    "skip_special_tokens": True,
                    "spaces_between_special_tokens": True,
                }
            )
        elif is_validate:
            request_sampling_params.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            )

        # Update with any additional kwargs
        request_sampling_params.update(kwargs)
        torch.cuda.synchronize()
        main_loop_start_time = time.time()
        while current_turns < self.config.multi_turn.max_assistant_turns:
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                torch.cuda.synchronize()
                pending_start_time = time.time()
                await self._handle_pending_state(_req)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
                torch.cuda.synchronize()
                pending_end_time = time.time()
                self.log_manager.log(
                    log_path,
                    event="pending_state_handling",
                    duration=pending_end_time - pending_start_time,
                    extra={"request_id": _req.request_id},
                    workid=self._rank,
                    step=self.step
                )
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                torch.cuda.synchronize()
                tool_calling_start_time = time.time()
                if _req.messages[-1].tool_calls is not None:
                    # === 工具调用相关变量 ===
                    # parsed_tool_calls: list[OpenAIFunctionToolCall] - 已解析的工具调用列表
                    #   每个 OpenAIFunctionToolCall 包含：
                    #     - id: str - 工具调用的唯一 ID
                    #     - function.name: str - 工具名称（如 "search_tool"）
                    #     - function.arguments: dict - 工具参数（JSON 格式）
                    parsed_tool_calls = _req.messages[-1].tool_calls
                    torch.cuda.synchronize()
                    tool_execution_start_time = time.time()

                    # 1017update: tool_call 增加原始query字段信息
                    # print("[DEBUG] tool_calling all: ", _req)
                    
                    # === 准备工具调用参数 ===
                    # prompt_str: str - 将初始 prompt token 序列解码为文本，用于提取用户原始问题
                    #   用途：在 search_tool 中通过 extract_prompt_str() 提取用户问题
                    #   注意：使用 prompt_ids 而不是 input_ids，因为我们需要的是原始问题，不是完整对话历史
                    prompt_str = self.processing_class.decode(_req.prompt_ids[0], skip_special_tokens=True)
                    # print("[DEBUG] tool_calling_prompt_str: ", prompt_str)
                    
                    # === 工具执行结果 ===
                    # tool_call_results: list[tuple[ToolResponse, float, dict]]
                    #   每个 tuple 包含三个元素：
                    #     - resp: ToolResponse - 工具返回的响应对象
                    #     - reward: float - 工具执行的即时奖励（当前未使用）
                    #     - tool_metrics: dict - 工具执行的指标
                    #         例如: {"total_results": 5, "status": "success", "api_request_error": None, ...}
                    try:
                        tool_call_results = await asyncio.gather(
                            *[
                                self._tool_map[tool_call.function.name].execute(
                                    _req.request_id,
                                    tool_call.function.arguments,
                                    **_req.tools_kwargs[tool_call.function.name].get("execute_kwargs", {"prompt_str": prompt_str}),
                                )
                                for tool_call in parsed_tool_calls
                            ]
                        )
                    except Exception as e:
                        # 工具执行失败 - 统一处理
                        logger.error(
                            f"[sglang_rollout._async_rollout_a_request] Request {_req.request_id} turn {current_turns}: "
                            f"Tool execution failed - {e}"
                        )
                        traceback.print_exc()
                        _req.search_error = True
                        _req.state = AsyncRolloutRequestStateEnum.FAILED
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    
                    # ========================================
                    # === 检测 search_error (环境异常) ===
                    # ========================================
                    # 从 tool_metrics 中检测搜索异常：
                    # 1. total_results == 0: 搜索无结果
                    # 2. status != "success": API调用失败或无结果
                    # 3. api_request_error is not None: API请求错误
                    for tool_result_idx, (_, _, tool_metrics) in enumerate(tool_call_results):
                        # doc_num: int - 本次工具调用返回的文档数量
                        doc_num = tool_metrics.get("total_results", 0)
                        current_docs += doc_num  # 累加到总文档计数
                        
                        # 检查搜索异常：结果为空或API错误
                        if doc_num == 0 or tool_metrics.get("status") != "success" or tool_metrics.get("api_request_error"):
                            _req.search_error = True
                            logger.warning(
                                f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                f"Search error detected - total_results={doc_num}, status={tool_metrics.get('status')}, "
                                f"api_error={tool_metrics.get('api_request_error')}"
                            )


                    # Extract search_call_stats from tool metrics for timing analysis
                    turn_search_call_stats = None
                    for _, _, t_metrics in tool_call_results:
                        if t_metrics and "search_call_stats" in t_metrics:
                            turn_search_call_stats = t_metrics["search_call_stats"]
                            break  # Take the first one (typically only one search tool per turn)

                    torch.cuda.synchronize()
                    tool_execution_end_time = time.time()
                    self.log_manager.log(
                        log_path,
                        event="tool_execution",
                        duration=tool_execution_end_time - tool_execution_start_time,
                        extra={"request_id": _req.request_id, "turn": current_turns,
                               "search_call_stats": turn_search_call_stats},
                        workid=self._rank,
                        step=self.step
                    )
                    torch.cuda.synchronize()
                    tool_response_start_time = time.time()
                    # === 添加工具响应到对话历史 ===
                    # 调用 add_tool_response_messages() 会：
                    # 1. 将 tool response 添加到 _req.messages
                    # 2. 通过 _update_input_ids() 更新 _req.input_ids
                    # 
                    # Token 序列变化：
                    # 之前: [sys, user, assistant(think+tool_call)]
                    # 之后: [sys, user, assistant(think+tool_call), tool(obs)]
                    # 
                    # 三者状态：
                    # - prompt_ids: 不变，仍为 [sys, user, <im_start>assistant\n]
                    # - input_ids: 增长，新增 tool(obs) 的 token
                    # - response_ids: 仍为 None（尚未 finalize）
                    try:
                        _req.add_tool_response_messages(self.processing_class, [resp for resp, _, _ in tool_call_results])
                        for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results, strict=True):
                            _req.update_metrics(metrics, tool_call.function.name)
                    except Exception as e:
                        # add_tool_response_messages 或 update_metrics 失败 - 统一处理
                        logger.error(
                            f"[sglang_rollout._async_rollout_a_request] Request {_req.request_id} turn {current_turns}: "
                            f"Failed to process tool responses - {e}"
                        )
                        traceback.print_exc()
                        _req.state = AsyncRolloutRequestStateEnum.FAILED
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        _req.tool_parse_error = True
                        break
                    torch.cuda.synchronize()
                    tool_response_end_time = time.time()
                    self.log_manager.log(
                        log_path,
                        event="tool_response_processing",
                        duration=tool_response_end_time - tool_response_start_time,
                        extra={"request_id": _req.request_id, "turn": current_turns},
                        workid=self._rank,
                        step=self.step
                    )
                    # ========================================
                    # === 工具调用后长度检查（检查点1）===
                    # ========================================
                    # 
                    # 在添加 tool response 到 input_ids 后，检查序列长度是否超限
                    # 
                    # 为什么要检查：
                    # - tool response 可能很长（如搜索返回大量文本）
                    # - 添加后可能导致序列超过 max_model_len
                    # - 必须在下一轮生成前检查，避免SGLang报错
                    # 
                    # 检查条件：len(_req.input_ids) >= self.config.max_model_len
                    # 
                    # 触发场景示例：
                    # - max_model_len = 8000
                    # - 当前 input_ids = 7900 tokens
                    # - tool response = 150 tokens
                    # - 新的 input_ids = 8050 tokens > 8000
                    # 
                    # 处理方式：
                    # - 设置 finish_reason = STOP
                    # - break 退出循环，结束生成
                    # - 不再尝试下一轮生成
                    # 
                    # 记录情况：
                    # - 打印 DEBUG 信息："FinishReasonTypeEnum.STOP current_turns={}, prompt_length={}"
                    # - event="async_rollout_request_complete"
                    # - extra={"finish_reason": "STOP", "total_sequence_length": len(input_ids)}
                    # print(f"[DEBUG]toolcall: {_req.messages}")
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        print(f"[DEBUG] FinishReasonTypeEnum.STOP current_turns{current_turns}, prompt_length{len(_req.input_ids)}")
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    # ========================================
                    # === 异常处理：状态不一致 ===
                    # ========================================
                    # 触发条件：
                    # - 状态为 TOOL_CALLING，但 messages[-1].tool_calls 为 None
                    # - 这是一个状态不一致的情况，可能由以下原因导致：
                    #   1. 代码逻辑错误
                    #   2. 异常处理路径中状态未正确更新
                    #
                    # 处理方式：
                    # - 记录详细的调试信息
                    # - 标记为 tool_parse_error
                    # - 设置状态为 COMPLETED 并 break
                    logger.error(
                        f"[sglang_rollout._async_rollout_a_request] Request {_req.request_id} turn {current_turns}: "
                        f"Unexpected state: TOOL_CALLING but messages[-1].tool_calls is None. "
                        f"Last message: {_req.messages[-1]}"
                    )
                    _req.tool_parse_error = True
                    _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                    finish_reason_type = FinishReasonTypeEnum.STOP
                    break
                torch.cuda.synchronize()
                tool_calling_end_time = time.time()
                self.log_manager.log(
                    log_path,
                    event="tool_calling_state",
                    duration=tool_calling_end_time - tool_calling_start_time,
                    extra={"request_id": _req.request_id, "turn": current_turns},
                    workid=self._rank,
                    step=self.step
                )
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                # === TURN START ===
                torch.cuda.synchronize()
                turn_start_time = time.time()
                self.log_manager.log(
                    log_path,
                    event="turn_start",
                    extra={
                        "request_id": _req.request_id,
                        "turn": current_turns,
                        "current_sequence_length": len(_req.get_generation_prompt_ids(self.processing_class))
                    },
                    workid=self._rank,
                    step=self.step
                )

                # === TURN PRE-PROCESS ===
                torch.cuda.synchronize()
                turn_pre_process_start_time = time.time()

                # ========================================
                # === 生成前长度检查（检查点2）===
                # ========================================
                # 
                # 在调用 engine 生成前，检查当前序列长度
                # 
                # 为什么要检查：
                # - SGLang 在生成时会自动添加 EOS token
                # - 如果 prompt_length + max_new_tokens + 1 > max_model_len，会报错
                # - 必须预留 1 个 token 位置给 EOS
                # 
                # 检查条件：prompt_length + 1 >= max_model_len
                # 
                # prompt_length 的含义：
                # - 通过 _req.get_generation_prompt_ids(self.processing_class) 获取
                # - 包含完整的对话历史（prompt + 所有已生成内容）
                # - 这是传递给 engine 的实际 token 序列
                # 
                # 触发场景示例：
                # - max_model_len = 8000
                # - prompt_length = 7999
                # - 7999 + 1 >= 8000 → 触发检查
                # 
                # 处理方式：
                # - 设置 finish_reason = LENGTH
                # - break 退出循环，结束生成
                # - 不调用 engine 生成
                # 
                # 与检查点1的区别：
                # - 检查点1：检查 len(_req.input_ids)（完整序列）
                # - 检查点2：检查 len(generation_prompt_ids)（用于生成的prompt）
                # - 两者通常相同，但在某些情况下可能略有差异
                # 
                # 记录情况：
                # - 打印 DEBUG 信息："FinishReasonTypeEnum.LENGTH current_turns={}, prompt_length={}"
                # - event="async_rollout_request_complete"
                # - extra={"finish_reason": "LENGTH"}
                # 
                # Only continue the conversation if the prompt length is not greater than max_model_len - 1,
                # since SGLang raises an error when max_new_tokens + 1 is greater to max_model_len (the extra
                # token accounts for the EOS token).
                prompt_length = len(_req.get_generation_prompt_ids(self.processing_class))

                if prompt_length + 1 >= self.config.max_model_len:
                    finish_reason_type = FinishReasonTypeEnum.LENGTH
                    print(f"[DEBUG] FinishReasonTypeEnum.LENGTH current_turns{current_turns}, prompt_length{prompt_length}")
                    break

                # Video support is not implemented yet
                image_data = (
                    _req.multi_modal_data["image"]
                    if _req.multi_modal_data and "image" in _req.multi_modal_data
                    else None
                )
                video_data = (
                    _req.multi_modal_data["video"]
                    if _req.multi_modal_data and "video" in _req.multi_modal_data
                    else None
                )
                if video_data:
                    logger.warning(
                        "video support is not implemented yet, current length of video data is %d", len(video_data)
                    )
                torch.cuda.synchronize()
                turn_pre_process_end_time = time.time()
                turn_pre_process_duration = turn_pre_process_end_time - turn_pre_process_start_time
                self.log_manager.log(
                    log_path,
                    event="turn_pre_process",
                    duration=turn_pre_process_duration,
                    extra={
                        "request_id": _req.request_id,
                        "turn": current_turns,
                        "generation_prompt_length": len(_req.get_generation_prompt_ids(self.processing_class))
                    },
                    workid=self._rank,
                    step=self.step
                )

                # === TURN ENGINE CALL ===
                torch.cuda.synchronize()
                turn_engine_call_start_time = time.time()
                # output: dict - 引擎生成的输出，包含以下关键字段：
                #   {
                #     "text": str - 生成的文本内容（包含 think + tool_call 或 final answer）
                #     "output_ids": list[int] - 生成的 token IDs
                #     "meta_info": {
                #       "finish_reason": {"type": str} - 结束原因（"stop"/"length"）
                #       "output_token_logprobs": list[tuple] - 每个 token 的 log probability
                #       ...
                #     }
                #   }
                output = await self._handle_engine_call(_req, request_sampling_params, image_data=image_data)
                torch.cuda.synchronize()
                turn_engine_call_end_time = time.time()
                turn_engine_call_duration = turn_engine_call_end_time - turn_engine_call_start_time
                self.log_manager.log(
                    log_path,
                    event="turn_engine_call",
                    duration=turn_engine_call_duration,
                    extra={
                        "request_id": _req.request_id,
                        "turn": current_turns,
                        "generated_tokens": len(output.get("output_ids", [])) if output else 0,
                        "output_text_length": len(output.get("text", "")) if output else 0
                    },
                    workid=self._rank,
                    step=self.step
                )

                # === TURN POST-PROCESS ===
                torch.cuda.synchronize()
                turn_post_process_start_time = time.time()

                # content: str - 从 output 中提取的生成文本
                #   格式示例: "<think>推理过程</think>\n<tool_call>...</tool_call>" 或纯文本答案
                content = output["text"]
                # finish_reason_type: FinishReasonTypeEnum - 引擎返回的结束原因
                #   可能值: STOP（正常结束）/ LENGTH（超长度）/ TOOL_CALL（需要工具调用）
                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])

                # Initialize tool parsing timing
                tool_parsing_duration = 0
                has_tool_calls = False
                parsed_tool_calls_count = 0
                turn_search_call_stats = None
                tool_execution_start_time = 0
                tool_execution_end_time = 0

                # 更新 assistant turns 计数（每次引擎生成后）
                current_turns += 1
                _req.total_assistant_turns = current_turns
                
                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(self.processing_class, content)
                    print(f"[DEBUG] FinishReasonTypeEnum.LENGTH current_turns{current_turns}")
                    break
                else:
                    if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                        finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                        _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                        torch.cuda.synchronize()
                        tool_parsing_start_time = time.time()
                        
                        # ========================================
                        # === 统计 <tool_call> 标签出现次数 ===
                        # ========================================
                        # 统计当前轮生成内容中有多少个 <tool_call> 标签
                        # 即使这些标签解析失败，也要计数（用于检测是否超过最大允许次数）
                        tool_call_tag_count = content.count("<tool_call>")
                        _req.total_tool_call_attempts += tool_call_tag_count
                        
                        # 初始化解析错误标记
                        has_parse_error = False
                        parse_error_reason = None
                        
                        try:
                            # ========================================
                            # === 工具调用解析（可能发生格式错误）===
                            # ========================================
                            # 
                            # self._function_call_parser.parse_non_stream(content) 尝试从生成的文本中解析工具调用
                            # 
                            # 输入: content - 模型生成的原始文本
                            #   示例: "<think>推理过程</think>\n<tool_call>{\"name\": \"search\","arguments":{"query_list": }}</tool_call>"
                            # 
                            # 输出:
                            #   - normed_content: str - 规范化后的内容（移除工具调用标签，保留think）
                            #   - tool_calls: list[ToolCallItem] - 原始工具调用列表
                            #       每个 ToolCallItem 包含:
                            #         - name: str - 工具名称（如 "search"）
                            #         - parameters: str - JSON格式的参数字符串
                            #         - tool_index: int - 工具调用索引
                            normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
                        except JSONDecodeError as e:
                            # ⚠️ 异常处理1：JSON解析失败
                            # 
                            # 触发条件：
                            # 1. <tool_call> 标签未正确闭合
                            #    示例: "<tool_call>{\"query\": \"test\"</tool_call>"（JSON未闭合）
                            # 2. JSON格式错误
                            #    示例: "<tool_call>{query: test}</tool_call>"（缺少引号）
                            # 3. <think> 标签未闭合
                            #    示例: "<think>推理中..."（缺少</think>）
                            # 
                            # 处理方式：
                            # - 保留原始内容（normed_content = content）
                            # - 清空工具调用列表（tool_calls = []）
                            # - 后续会将此内容作为普通文本添加到messages
                            # - 设置 finish_reason = STOP，结束生成
                            # 
                            # 记录情况：
                            # - event="turn_tool_parsing"
                            # - extra={"raw_tool_calls": 0, "parsed_tool_calls": 0}
                            normed_content = content
                            tool_calls = []
                            has_parse_error = True
                            parse_error_reason = f"JSONDecodeError: {str(e)}"
                            _req.tool_parse_error = True
                            logger.warning(f"[Abnormal] Request {_req.request_id} turn {current_turns}: JSON parse error - {str(e)}")
                        except AttributeError as e:
                            # ⚠️ 异常处理2：Parser属性错误
                            # 
                            # 触发条件：
                            # - Parser对象缺少必要的属性或方法（极少见）
                            # 
                            # 处理方式：同 JSONDecodeError
                            normed_content = content
                            tool_calls = []
                            has_parse_error = True
                            parse_error_reason = f"AttributeError: {str(e)}"
                            _req.tool_parse_error = True
                            logger.warning(f"[Abnormal] Request {_req.request_id} turn {current_turns}: Parser attribute error - {str(e)}")
                        # ========================================
                        # === 工具调用参数验证和过滤 ===
                        # ========================================
                        # 
                        # 工具格式：<tool_call>{"name": "search", "arguments": {"query_list": ["q1","q2","q3"]}}</tool_call>
                        # 
                        # 从原始 tool_calls 列表中过滤出有效的工具调用
                        # 过滤条件：
                        # 1. 工具名称必须是 "search"（在 tool_map 中）
                        # 2. 参数必须能正确解析为 dict 格式
                        # 3. 参数中必须有 query_list 字段
                        
                        parsed_tool_calls = []
                        invalid_tool_calls_count = 0  # 参数解析失败的工具调用数
                        unknown_tool_calls_count = 0  # 未知工具的调用数
                        unknown_tool_names = []  # 记录未知工具的名称（用于监控）
                        
                        # 提取所有查询用于重复检测和统计
                        current_turn_queries = []  # 当前轮次所有的 query
                        current_turn_query_count = 0  # 当前轮次的 query 总数（query_list 的总长度）
                        
                        for tool_call in tool_calls:
                            # ========================================
                            # === 检测未知工具调用 ===
                            # ========================================
                            # 检查工具是否在工具映射中（应该只有 "search"）
                            if tool_call.name not in self._tool_map:
                                unknown_tool_calls_count += 1
                                unknown_tool_names.append(tool_call.name)
                                has_parse_error = True
                                if parse_error_reason is None:
                                    parse_error_reason = f"Unknown tool: {tool_call.name}"
                                _req.tool_parse_error = True
                                logger.warning(
                                    f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                    f"Unknown tool '{tool_call.name}' called (expected 'search')"
                                )
                                continue  # 跳过未知工具
                            
                            # ========================================
                            # === 参数验证：解析 JSON 字符串 ===
                            # ========================================
                            # 尝试将 JSON 字符串转换为 dict
                            # 
                            # OpenAIFunctionCallSchema.from_openai_function_parsed_schema() 执行：
                            # 1. 尝试 json.loads(tool_call.parameters) 解析 JSON
                            # 2. 检查解析结果是否为 dict 类型
                            # 3. 返回 (function, has_decode_error)
                            function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                                OpenAIFunctionParsedSchema(
                                    name=tool_call.name,
                                    arguments=tool_call.parameters,  # JSON字符串，如 '{"query_list": ["q1","q2"]}'
                                )
                            )
                            
                            # ⚠️ 异常处理：参数解码错误
                            # 触发条件：has_decode_error = True
                            # - JSON解析失败（格式错误、未闭合括号等）
                            # - 解析结果不是 dict 类型
                            # - 参数为空或格式错误
                            if has_decode_error:
                                invalid_tool_calls_count += 1
                                has_parse_error = True
                                if parse_error_reason is None:
                                    parse_error_reason = f"Parameter decode error for tool: {tool_call.name}"
                                _req.tool_parse_error = True
                                logger.warning(
                                    f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                    f"Parameter decode error for tool '{tool_call.name}'"
                                )
                                continue  # 丢弃有错误的工具调用
                            
                            # ========================================
                            # === 提取 query_list 用于统计和重复检测 ===
                            # ========================================
                            # 工具参数格式：{"query_list": ["subq1", "subq2", "subq3"]}
                            try:
                                if isinstance(function.arguments, dict) and 'query_list' in function.arguments:
                                    query_list = function.arguments['query_list']
                                    if isinstance(query_list, list):
                                        # 使用 extend 添加所有查询到当前轮次的查询列表
                                        valid_queries = [q for q in query_list if isinstance(q, str) and q.strip()]
                                        current_turn_queries.extend(valid_queries)
                                        current_turn_query_count += len(valid_queries)
                                    else:
                                        logger.warning(
                                            f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                            f"query_list is not a list: {type(query_list)}"
                                        )
                                else:
                                    logger.warning(
                                        f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                        f"query_list not found in tool arguments"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                    f"Failed to extract query_list: {e}"
                                )
                            
                            parsed_tool_calls.append(
                                OpenAIFunctionToolCall(
                                    id=str(tool_call.tool_index),
                                    function=function,
                                )
                            )
                        
                        parsed_tool_calls_count = len(parsed_tool_calls)  # int: 成功解析的工具调用数量
                        
                        # ========================================
                        # === 检测1：单轮 query 数量超限 ===
                        # ========================================
                        # 检查 query_list 的总长度是否超过限制（默认5个）
                        max_queries_per_turn = self.config.multi_turn.get('max_tool_calls_per_turn', 5)
                        if current_turn_query_count > max_queries_per_turn:
                            _req.excessive_tool_calls_per_turn = True
                            logger.warning(
                                f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                f"Excessive queries {current_turn_query_count} > {max_queries_per_turn}"
                            )
                        
                        # ========================================
                        # === 检测2：重复查询检测 ===
                        # ========================================
                        has_repeated_query = False
                        repeated_queries = []
                        for query in current_turn_queries:
                            if _req.check_repeated_query(query):
                                _req.repeated_query = True
                                has_repeated_query = True
                                repeated_queries.append(query)
                                logger.warning(
                                    f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                    f"Repeated query detected: '{query}'"
                                )
                        
                        # 将当前轮次的所有查询添加到历史记录（使用 extend）
                        _req.query_history.extend(current_turn_queries)
                        
                        # 将未知工具名称添加到 _req 中（用于后续监控和分析）
                        if unknown_tool_names:
                            _req.unknown_tool_names.extend(unknown_tool_names)
                        
                        # ========================================
                        # === 更新统计指标 ===
                        # ========================================
                        # 记录本轮 query 数量（query_list 的总长度）
                        _req.tool_calls_per_turn.append(current_turn_query_count)
                        # 更新总 query 数量
                        _req.total_tool_calls += current_turn_query_count
                        # 更新单轮最大 query 数量
                        if current_turn_query_count > _req.max_tool_calls_in_single_turn:
                            _req.max_tool_calls_in_single_turn = current_turn_query_count
                        
                        torch.cuda.synchronize()
                        tool_parsing_end_time = time.time()
                        tool_parsing_duration = tool_parsing_end_time - tool_parsing_start_time
                        self.log_manager.log(
                            log_path,
                            event="turn_tool_parsing",
                            duration=tool_parsing_duration,
                            extra={
                                "request_id": _req.request_id,
                                "turn": current_turns,
                                "raw_tool_calls": len(tool_calls),  # 原始解析到的工具调用数量
                                "parsed_tool_calls": parsed_tool_calls_count,  # 成功解析的工具调用数量
                                "invalid_tool_calls": invalid_tool_calls_count,  # 参数解析失败的数量
                                "unknown_tool_calls": unknown_tool_calls_count,  # 未知工具的数量
                                "unknown_tool_names": unknown_tool_names,  # 未知工具名称列表
                                "current_turn_query_count": current_turn_query_count,  # 本轮 query 总数
                                "current_turn_queries": current_turn_queries,  # 本轮所有 query
                                "has_parse_error": has_parse_error,
                                "parse_error_reason": parse_error_reason,
                                "has_repeated_query": has_repeated_query,
                                "repeated_queries": repeated_queries,  # 重复的 query 列表
                                "excessive_queries": _req.excessive_tool_calls_per_turn,
                            },
                            workid=self._rank,
                            step=self.step
                        )

                        # ========================================
                        # === Break + 0 reward 异常检测 ===
                        # ========================================
                        # 核心逻辑：
                        # 只要模型输出了 <tool_call>（即 has_tool_call(content) 返回 True），
                        # 就说明模型想要调用工具。
                        # 
                        # 此时如果没有成功解析出有效的工具调用，或者有其他异常，
                        # 都应该 Break + 0 reward。
                        # 
                        # 异常情况包括：
                        # 1. parsed_tool_calls_count == 0：没有成功解析出任何工具调用
                        #    - JSON 解析失败（tool_parse_error）
                        #    - 所有工具都是未知工具（unknown_tool_calls）
                        #    - 参数解码失败（invalid_tool_calls）
                        # 2. excessive_tool_calls_per_turn：query_list 长度超限
                        # 3. repeated_query：有重复查询
                        
                        # 检查是否没有成功解析出工具调用
                        no_valid_tool_calls = (parsed_tool_calls_count == 0)
                        
                        # 判断是否需要 Break + 0 reward
                        should_break_with_zero_reward = (
                            no_valid_tool_calls or  # 没有成功解析出工具调用（最关键）
                            _req.tool_parse_error # or  # 解析错误
                            # _req.excessive_tool_calls_per_turn or  # query 数量超限
                            # _req.repeated_query  # 重复查询
                        )
                        
                        if should_break_with_zero_reward:
                            # 将内容作为普通文本添加（不带tool_calls）
                            _req.add_assistant_message(self.processing_class, content)
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            
                            # 打印异常原因
                            abnormal_reasons = []
                            if no_valid_tool_calls:
                                abnormal_reasons.append(f"no_valid_tool_calls(parsed={parsed_tool_calls_count})")
                            if _req.tool_parse_error:
                                abnormal_reasons.append(f"tool_parse_error({parse_error_reason})")
                            if _req.excessive_tool_calls_per_turn:
                                abnormal_reasons.append(f"excessive_queries({current_turn_query_count}>{max_queries_per_turn})")
                            if _req.repeated_query:
                                abnormal_reasons.append(f"repeated_query({len(repeated_queries)} duplicates)")
                            
                            logger.warning(
                                f"[Abnormal Break] Request {_req.request_id} turn {current_turns}: "
                                f"Breaking with zero reward due to: {', '.join(abnormal_reasons)}"
                            )
                            print(
                                f"[DEBUG] FinishReasonTypeEnum.STOP (Abnormal Break): {', '.join(abnormal_reasons)}"
                            )
                            break
                        
                        # 如果走到这里，说明有有效的工具调用，且没有异常
                        if len(parsed_tool_calls) > 0:
                            # ===  有有效的工具调用 ===
                            # 
                            # 将 assistant message（包含 tool_calls）添加到对话历史
                            # Token 序列变化：
                            # 之前: [..., tool(obs_n-1)] 或 [sys, user]
                            # 之后: [..., assistant(think+tool_call)]
                            # 
                            # 三者状态：
                            # - prompt_ids: 不变
                            # - input_ids: 增长，新增 assistant(think+tool_call) 的 token
                            # - response_ids: 仍为 None
                            _req.add_assistant_message(
                                self.processing_class, normed_content, tool_calls=parsed_tool_calls
                            )
                        else:
                            # ⚠️ 异常处理4：没有有效的工具调用
                            # 
                            # 触发条件：
                            # 1. 模型输出了空的工具调用
                            #    示例: "<tool_call></tool_call>"
                            # 2. 所有工具调用都被过滤（has_decode_error = True）
                            #    示例: 3个工具调用都有参数错误，全部被丢弃
                            # 3. parse_non_stream() 返回空列表（JSONDecodeError后）
                            # 
                            # 处理方式：
                            # - 将原始内容作为普通文本添加到messages（不带tool_calls）
                            # - 设置 finish_reason = STOP
                            # - 设置 state = COMPLETED
                            # - break 退出循环，结束生成
                            # 
                            # Token 序列变化：
                            # 之前: [..., tool(obs_n-1)]
                            # 之后: [..., assistant(content)]  # 作为普通文本
                            # 
                            # 后果：
                            # - 此轮生成被视为最终答案
                            # - 即使内容包含格式错误的工具调用，也会被保留在对话中
                            # - 模型无法继续调用工具
                            # 
                            # 记录情况：
                            # - 打印 DEBUG 信息："FinishReasonTypeEnum.STOP 没有tool_calls"
                            # - event="turn_end"
                            # - extra={"finish_reason": "STOP", "has_tool_calls": False}
                            _req.add_assistant_message(self.processing_class, content)
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            print(f"[DEBUG] FinishReasonTypeEnum.STOP 没有tool_calls")
                            break
                    else:
                        # === 添加 assistant 消息（最终答案，无 tool_calls）===
                        # 这是最后一轮生成，assistant 给出最终答案
                        # Token 序列变化：
                        # 之前: [..., tool(obs_n)]
                        # 之后: [..., tool(obs_n), assistant(final_answer)]
                        # 
                        # 三者状态：
                        # - prompt_ids: 不变
                        # - input_ids: 增长，新增 assistant(final_answer) 的 token
                        # - response_ids: 仍为 None（将在 finalize 时计算）
                        _req.add_assistant_message(
                            self.processing_class,
                            content,
                        )
                        if (
                            _req.interaction_kwargs
                            and self.interaction_map
                            and user_turns < self.config.multi_turn.max_user_turns
                            and current_turns < self.config.multi_turn.max_assistant_turns
                        ):
                            _req.state = AsyncRolloutRequestStateEnum.INTERACTING
                        else:
                            break


                torch.cuda.synchronize()
                turn_post_process_end_time = time.time()
                turn_post_process_duration = turn_post_process_end_time - turn_post_process_start_time
                self.log_manager.log(
                    log_path,
                    event="turn_post_process",
                    duration=turn_post_process_duration,
                    extra={
                        "request_id": _req.request_id,
                        "turn": current_turns,
                        "has_tool_calls": has_tool_calls,
                        "parsed_tool_calls": parsed_tool_calls_count,
                        "finish_reason": str(finish_reason_type),
                        "tool_parsing_duration": tool_parsing_duration
                    },
                    workid=self._rank,
                    step=self.step
                )

                # === TURN END ===
                current_turns += 1
                torch.cuda.synchronize()
                turn_end_time = time.time()
                turn_total_duration = turn_end_time - turn_start_time

                # Record turn timing details
                turn_timing = {
                    "turn": current_turns - 1,
                    "total_duration": turn_total_duration,
                    "pre_process_duration": turn_pre_process_duration,
                    "engine_call_duration": turn_engine_call_duration,
                    "post_process_duration": turn_post_process_duration,
                    "tool_parsing_duration": tool_parsing_duration,
                    "tool_execution_duration": (tool_execution_end_time - tool_execution_start_time) if has_tool_calls else 0.0,
                    "has_tool_calls": has_tool_calls,
                    "finish_reason": str(finish_reason_type),
                    "search_call_stats": turn_search_call_stats if has_tool_calls else None,
                }
                turn_timings.append(turn_timing)

                self.log_manager.log(
                    log_path,
                    event="turn_end",
                    duration=turn_total_duration,
                    extra={
                        "request_id": _req.request_id,
                        "turn": current_turns - 1,
                        "pre_process_duration": turn_pre_process_duration,
                        "engine_call_duration": turn_engine_call_duration,
                        "post_process_duration": turn_post_process_duration,
                        "tool_parsing_duration": tool_parsing_duration,
                        "has_tool_calls": has_tool_calls,
                        "finish_reason": str(finish_reason_type),
                        "engine_call_pct": round(turn_engine_call_duration / turn_total_duration * 100, 2),
                        "post_process_pct": round(turn_post_process_duration / turn_total_duration * 100, 2)
                    },
                    workid=self._rank,
                    step=self.step
                )




            elif _req.state == AsyncRolloutRequestStateEnum.INTERACTING:
                torch.cuda.synchronize()
                interacting_start_time = time.time()
                user_turns += 1
                messages = [{"role": x.role, "content": x.content} for x in _req.messages]

                # Get interaction by name from interaction_kwargs
                interaction_name = _req.interaction_kwargs.get(
                    "name", "gsm8k"
                )  # Default to gsm8k for backward compatibility
                if interaction_name not in self.interaction_map:
                    raise ValueError(
                        f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                        f"{list(self.interaction_map.keys())}"
                    )

                interaction = self.interaction_map[interaction_name]
                torch.cuda.synchronize()
                interaction_response_start_time = time.time()
                should_terminate_sequence, content, reward, metrics = await interaction.generate_response(
                    _req.request_id, messages, **_req.interaction_kwargs
                )
                torch.cuda.synchronize()
                interaction_response_end_time = time.time()
                self.log_manager.log(
                    log_path,
                    event="interaction_response",
                    duration=interaction_response_end_time - interaction_response_start_time,
                    extra={
                        "request_id": _req.request_id,
                        "user_turn": user_turns,
                        "assistant_turn": current_turns
                    },
                    workid=self._rank,
                    step=self.step
                )
                user_turn_rewards.append(reward)
                if should_terminate_sequence:
                    finish_reason_type = FinishReasonTypeEnum.STOP
                    _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                    break
                else:
                    _req.add_user_message(self.processing_class, content)
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    else:
                        _req.state = AsyncRolloutRequestStateEnum.RUNNING
                torch.cuda.synchronize()
                interacting_end_time = time.time()
                self.log_manager.log(
                    log_path,
                    event="interacting_state",
                    duration=interacting_end_time - interacting_start_time,
                    extra={
                        "request_id": _req.request_id,
                        "user_turn": user_turns,
                        "assistant_turn": current_turns
                    },
                    workid=self._rank,
                    step=self.step
                )
        torch.cuda.synchronize()
        main_loop_end_time = time.time()

        self.log_manager.log(
            log_path,
            event="main_loop",
            duration=main_loop_end_time - main_loop_start_time,
            extra={"request_id": _req.request_id},
            workid=self._rank,
            step=self.step
        )

        # ========================================
        # === 超出最大轮次检查（循环后）===
        # ========================================
        # 
        # 在主循环结束后，检查是否因为达到最大轮次而退出
        # 
        # 触发条件：current_turns >= self.config.multi_turn.max_assistant_turns
        # 
        # 默认配置：
        # - max_assistant_turns = max_model_len // 3（如果未配置）
        # - 例如：max_model_len=8000 → max_assistant_turns=2666
        # 
        # 循环机制：
        # - 循环条件：while current_turns < max_assistant_turns
        # - 每次生成后：current_turns += 1
        # - 当 current_turns >= max_assistant_turns 时，循环自然退出
        # 
        # 处理方式：
        # - 设置 finish_reason = STOP
        # - 设置 exceed_max_turns 标记
        # - 继续后续的 finalize 流程（后续会给 0 reward）
        # 
        # 记录情况：
        # - 打印 DEBUG 信息："FinishReasonTypeEnum.STOP current_turns={} >= max_assistant_turns={}"
        # - event="async_rollout_request_complete"
        # - extra={"finish_reason": "STOP", "turns": current_turns, "exceed_max_turns": True}
        if _req.total_tool_call_attempts > 100:  # 硬编码最大允许的工具调用次数为100
            finish_reason_type = FinishReasonTypeEnum.STOP
            _req.exceed_max_turns = True
            logger.warning(
                f"[Abnormal] Request {_req.request_id}: "
                f"Exceeded max tool call attempts {_req.total_tool_call_attempts} > 100"
            )
            print(
                f"[DEBUG] FinishReasonTypeEnum.STOP (Exceed Max Tool Calls) "
                f"total_tool_call_attempts={_req.total_tool_call_attempts} > 100"
            )
        
        # ========================================
        # === 检查token数是否超限 ===
        # ========================================
        # 检查当前序列长度是否超过最大限制
        # 如果超过，标记 exceed_max_tokens=True
        # 这些样本：
        # - 仍用于计算组内 advantage（减少统计偏差）
        # - 但不参与 loss / 梯度回传（后续由 Trainer 处理）
        current_seq_len = len(_req.input_ids.squeeze(0)) if _req.input_ids is not None else 0
        if current_seq_len > self.config.max_model_len:
            _req.exceed_max_tokens = True
            logger.warning(
                f"[Abnormal] Request {_req.request_id}: "
                f"Exceeded max tokens {current_seq_len} > {self.config.max_model_len}"
            )
            print(
                f"[DEBUG] exceed_max_tokens: current_seq_len={current_seq_len} > "
                f"max_model_len={self.config.max_model_len}"
            )

        # Calculate the reward for each tool
        torch.cuda.synchronize()
        reward_calculation_start_time = time.time()
        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        tool_reward_tasks = []
        for name in _req.tools_kwargs.keys():
            tool = self._tool_map[name]
            tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
        tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = dict(tool_reward_scores)
        all_rewards = {**tool_reward_scores, **{"user_turn_rewards": user_turn_rewards}}
        torch.cuda.synchronize()
        reward_calculation_end_time = time.time()
        self.log_manager.log(
            log_path,
            event="reward_calculation",
            duration=reward_calculation_end_time - reward_calculation_start_time,
            extra={"request_id": _req.request_id},
            workid=self._rank,
            step=self.step
        )

        torch.cuda.synchronize()
        finalization_start_time = time.time()
        
        # 保存 finish_reason_type 到请求对象
        _req.finish_reason_type = finish_reason_type
        
        # === Finalize 请求：计算 response_ids ===
        # 调用 finalize() 会：
        # 1. 设置 state = COMPLETED
        # 2. 保存 reward_scores 和 finish_reason_type
        # 3. **计算 response_ids = input_ids[:, prompt_ids.shape[-1]:]**
        # 4. 执行 tokenization sanity check（对比完整 tokenize 和增量 tokenize 的结果）
        # 
        # 三者最终状态（Multi-turn Tool Use 完整示例）：
        # 
        # prompt_ids: [1, 50]
        #   内容: [sys, user, <im_start>assistant\n]
        #   说明: 原始 prompt，始终不变
        # 
        # input_ids: [1, 800]
        #   内容: [sys, user, 
        #          assistant(think1), assistant(tool_call1), tool(obs1),
        #          assistant(think2), assistant(tool_call2), tool(obs2),
        #          assistant(think3), assistant(final_answer)]
        #   说明: 完整对话序列，包含所有轮次
        # 
        # response_ids: [1, 750]  
        #   内容: [assistant(think1), assistant(tool_call1), tool(obs1),
        #          assistant(think2), assistant(tool_call2), tool(obs2),
        #          assistant(think3), assistant(final_answer)]
        #   说明: response_ids = input_ids[:, 50:] (从 prompt 结束位置开始切片)
        #         包含所有 agent 的推理、工具调用和观察结果
        # 
        # 对应的 loss_mask:
        # - prompt_loss_mask: [1, 50] 全为 0（不计算 loss）
        # - response_loss_mask: [1, 750]
        #     其中: assistant 生成的 token 为 1（计算 loss）
        #          tool observation 的 token 为 0（不计算 loss）
        #   示例: [1,1,1,...,1, 0,0,0,...,0, 1,1,1,...,1, 0,0,0,...,0, 1,1,1,...,1]
        #         ^assistant1   ^tool1       ^assistant2   ^tool2       ^assistant3
        _req.finalize(self.processing_class, all_rewards, finish_reason_type)
        torch.cuda.synchronize()
        finalization_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="finalization",
                duration=finalization_end_time - finalization_start_time,
                extra={"request_id": _req.request_id},
                workid=self._rank,
                step=self.step
            )

        torch.cuda.synchronize()
        request_end_time = time.time()
        total_request_time = request_end_time - request_start_time
        if self.config.calculate_log_probs:
            debug_sampling_params = {**self.sampling_params}
            debug_sampling_params["max_new_tokens"] = 0
            output = await self._engine.async_generate(
                prompt=None,
                input_ids=_req.input_ids,
                sampling_params=debug_sampling_params,
                return_logprob=True,
                logprob_start_len=0,
            )
            # len(input_token_logprobs) = len(input_tokens)-1，because logprob of 1st token is None
            _req.output_token_ids, _req.rollout_log_probs = _extract_logprob_from_output(output)


        response_length = len(_req.response_ids.squeeze(0)) if _req.response_ids is not None else 0
        actual_response_tokens = torch.sum(_req.response_loss_mask.squeeze(0)).item() if _req.response_loss_mask is not None else 0

        # Calculate turn statistics
        if turn_timings:
            avg_turn_duration = np.mean([t["total_duration"] for t in turn_timings])
            avg_engine_call_duration = np.mean([t["engine_call_duration"] for t in turn_timings])
            avg_post_process_duration = np.mean([t["post_process_duration"] for t in turn_timings])
            total_engine_time = sum([t["engine_call_duration"] for t in turn_timings])
            total_post_process_time = sum([t["post_process_duration"] for t in turn_timings])
            turns_with_tools = sum([1 for t in turn_timings if t["has_tool_calls"]])
            total_tool_execution_time = sum([t.get("tool_execution_duration", 0) for t in turn_timings])
        else:
            avg_turn_duration = 0
            avg_engine_call_duration = 0
            avg_post_process_duration = 0
            total_engine_time = 0
            total_post_process_time = 0
            turns_with_tools = 0
            total_tool_execution_time = 0

        # ========================================
        # === Aggregate search/refine stats across turns (方案B: step-level) ===
        # ========================================
        agg_search_stats = {
            "total_search_api_duration": 0.0,
            "total_dedup_duration": 0.0,
            "total_refine_duration": 0.0,
            "total_format_duration": 0.0,
            "total_search_call_duration": 0.0,
            "total_search_api_retries": 0,
            "total_refine_retries": 0,
            "refine_total_docs": 0,
            "refine_output_docs": 0,
            "refine_mode": "none",
            "refine_short_content_preserved": 0,
            "refine_refined_count": 0,
            "refine_error_count": 0,
            "refine_zero_relevance_filtered": 0,
            "refine_avg_compression_ratio": 0.0,
            "refine_avg_relevance": 0.0,
            "refine_status_distribution": {},
            "refine_relevance_distribution": {"0-0.1": 0, "0.2-0.4": 0, "0.5-0.7": 0, "0.8-1.0": 0},
        }
        refine_comp_weighted = 0.0
        refine_rel_weighted = 0.0
        refine_total_n = 0
        for tt in turn_timings:
            scs = tt.get("search_call_stats")
            if scs:
                agg_search_stats["total_search_api_duration"] += scs.get("search_api_duration", 0)
                agg_search_stats["total_dedup_duration"] += scs.get("dedup_duration", 0)
                agg_search_stats["total_refine_duration"] += scs.get("refine_duration", 0)
                agg_search_stats["total_format_duration"] += scs.get("format_duration", 0)
                agg_search_stats["total_search_call_duration"] += scs.get("total_call_duration", 0)
                agg_search_stats["total_search_api_retries"] += scs.get("search_api_retries", 0)
                rs = scs.get("refine_stats", {})
                if rs:
                    agg_search_stats["total_refine_retries"] += rs.get("refine_retries", 0)
                    n = rs.get("total_docs", 0)
                    agg_search_stats["refine_total_docs"] += n
                    agg_search_stats["refine_output_docs"] += rs.get("output_docs", 0)
                    agg_search_stats["refine_mode"] = rs.get("refine_mode", "none")
                    agg_search_stats["refine_short_content_preserved"] += rs.get("short_content_preserved_count", 0)
                    agg_search_stats["refine_refined_count"] += rs.get("refined_count", 0)
                    agg_search_stats["refine_error_count"] += rs.get("error_count", 0)
                    agg_search_stats["refine_zero_relevance_filtered"] += rs.get("zero_relevance_filtered_count", 0)
                    if n > 0 and rs.get("avg_compression_ratio", 0) > 0:
                        refine_comp_weighted += rs["avg_compression_ratio"] * n
                        refine_total_n += n
                    if n > 0 and rs.get("avg_relevance", 0) > 0:
                        refine_rel_weighted += rs["avg_relevance"] * n
                    for k, v in rs.get("status_distribution", {}).items():
                        agg_search_stats["refine_status_distribution"][k] = agg_search_stats["refine_status_distribution"].get(k, 0) + v
                    for k, v in rs.get("relevance_distribution", {}).items():
                        agg_search_stats["refine_relevance_distribution"][k] = agg_search_stats["refine_relevance_distribution"].get(k, 0) + v
        if refine_total_n > 0:
            agg_search_stats["refine_avg_compression_ratio"] = round(refine_comp_weighted / refine_total_n, 4)
            agg_search_stats["refine_avg_relevance"] = round(refine_rel_weighted / refine_total_n, 4)
        
        # ========================================
        # === 计算行为指标统计 ===
        # ========================================
        # 搜索步数统计（assistant消息数）
        search_steps = _req.total_assistant_turns
        
        # 搜索调用次数统计
        total_tool_calls = _req.total_tool_calls
        
        # 单轮工具调用数的统计
        if _req.tool_calls_per_turn:
            avg_tool_calls_per_turn = np.mean(_req.tool_calls_per_turn)
            median_tool_calls_per_turn = np.median(_req.tool_calls_per_turn)
            max_tool_calls_per_turn = np.max(_req.tool_calls_per_turn)
        else:
            avg_tool_calls_per_turn = 0
            median_tool_calls_per_turn = 0
            max_tool_calls_per_turn = 0

        self.log_manager.log(
            log_path,
            event="async_rollout_request_complete",
            duration=total_request_time,
            extra={
                "request_id": _req.request_id,
                "batch_data_id": _req.batch_data_id,
                "finish_reason": str(finish_reason_type),
                "turns": current_turns,
                "user_turns": user_turns,
                "turns_with_tools": turns_with_tools,
                "response_length": response_length,
                "actual_response_tokens": actual_response_tokens,
                "total_sequence_length": len(_req.input_ids.squeeze(0)) if _req.input_ids is not None else 0,
                "avg_turn_duration": round(avg_turn_duration, 4),
                "avg_engine_call_duration": round(avg_engine_call_duration, 4),
                "avg_post_process_duration": round(avg_post_process_duration, 4),
                "total_engine_time": round(total_engine_time, 4),
                "total_post_process_time": round(total_post_process_time, 4),
                "engine_time_pct": round(total_engine_time / total_request_time * 100, 2) if total_request_time > 0 else 0,
                "post_process_time_pct": round(total_post_process_time / total_request_time * 100, 2) if total_request_time > 0 else 0,
                "total_tool_execution_time": round(total_tool_execution_time, 4),
                "tool_execution_time_pct": round(total_tool_execution_time / total_request_time * 100, 2) if total_request_time > 0 else 0,
                "turn_timings": turn_timings,
                # ========================================
                # === Search/Refine 阶段统计 (step-level) ===
                # ========================================
                "search_refine_stats": agg_search_stats,
                # ========================================
                # === 异常标记字段 ===
                # ========================================
                "excessive_tool_calls_per_turn": _req.excessive_tool_calls_per_turn,
                "tool_parse_error": _req.tool_parse_error,
                "repeated_query": _req.repeated_query,
                "search_error": _req.search_error,
                "exceed_max_turns": _req.exceed_max_turns,
                "exceed_max_tokens": _req.exceed_max_tokens,
                # ========================================
                # === 行为指标统计 ===
                # ========================================
                "search_steps": search_steps,  # assistant消息总数
                "total_tool_calls": total_tool_calls,  # 工具调用总次数
                "avg_tool_calls_per_turn": round(avg_tool_calls_per_turn, 2),
                "median_tool_calls_per_turn": median_tool_calls_per_turn,
                "max_tool_calls_in_single_turn": _req.max_tool_calls_in_single_turn,
                "tool_calls_per_turn": _req.tool_calls_per_turn,  # 每轮工具调用数列表
                "query_history_length": len(_req.query_history),  # 查询历史数量
            },
            workid=self._rank,
            step=self.step
        )

        # ========================================
        # === Write step summary to debug log ===
        # ========================================
        try:
            _debug_log_dir = os.getenv("KNOWGPO_DEBUG_LOG_DIR", "logs/knowgpo_debug")
            os.makedirs(_debug_log_dir, exist_ok=True)
            _debug_log_path = os.path.join(_debug_log_dir, "step_summary.log")
            _summary_record = {
                "timestamp": datetime.now().isoformat(),
                "worker_id": self._rank,
                "step": self.step,
                "request_id": _req.request_id,
                "batch_data_id": _req.batch_data_id,
                "finish_reason": str(finish_reason_type),
                "turns": current_turns,
                "user_turns": user_turns,
                "turns_with_tools": turns_with_tools,
                "response_length": response_length,
                "total_duration": round(total_request_time, 4),
                "avg_turn_duration": round(avg_turn_duration, 4),
                "avg_engine_call_duration": round(avg_engine_call_duration, 4),
                "total_engine_time": round(total_engine_time, 4),
                "engine_time_pct": round(total_engine_time / total_request_time * 100, 2) if total_request_time > 0 else 0,
                "total_post_process_time": round(total_post_process_time, 4),
                "post_process_time_pct": round(total_post_process_time / total_request_time * 100, 2) if total_request_time > 0 else 0,
                "total_tool_execution_time": round(total_tool_execution_time, 4),
                "tool_execution_time_pct": round(total_tool_execution_time / total_request_time * 100, 2) if total_request_time > 0 else 0,
                "search_refine_stats": agg_search_stats,
                "search_steps": search_steps,
                "total_tool_calls": total_tool_calls,
                "avg_tool_calls_per_turn": round(avg_tool_calls_per_turn, 2),
                "median_tool_calls_per_turn": median_tool_calls_per_turn,
                "max_tool_calls_in_single_turn": _req.max_tool_calls_in_single_turn,
                "tool_calls_per_turn": _req.tool_calls_per_turn,
                "excessive_tool_calls_per_turn": _req.excessive_tool_calls_per_turn,
                "tool_parse_error": _req.tool_parse_error,
                "repeated_query": _req.repeated_query,
                "search_error": _req.search_error,
                "exceed_max_turns": _req.exceed_max_turns,
                "exceed_max_tokens": _req.exceed_max_tokens,
            }
            with open(_debug_log_path, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(_summary_record, ensure_ascii=False) + "\n")
                _f.flush()
        except Exception as _e:
            logger.warning(f"[step_summary] Failed to write debug log: {_e}")

        return _req

    async def _handle_engine_call(
        self, _req: AsyncRolloutRequest, sampling_params: dict, image_data: Optional[list[Any]] = None
    ) -> dict:
        generation_prompt_ids = _req.get_generation_prompt_ids(self.processing_class)
        return await self._handle_engine_generate(generation_prompt_ids, sampling_params, image_data)

    async def _handle_engine_generate(
        self, generation_prompt_ids: list[int], sampling_params: dict, image_data: Optional[list[Any]] = None, request_id: Optional[str] = None
    ) -> dict:
        torch.cuda.synchronize()
        setup_start_time = time.time()
        max_new_tokens = min(self.config.response_length, self.config.max_model_len - len(generation_prompt_ids) - 1)

        kwargs = sampling_params.copy()
        kwargs["max_new_tokens"] = max_new_tokens
        kwargs["n"] = 1  # group size is supported in preprocess
        torch.cuda.synchronize()
        setup_end_time = time.time()
        log_path = os.path.join(
            self.log_dir,
            f"step_{self.step}",
            f"worker_{self._rank}.jsonl"
        )
        self.log_manager.log(
            log_path,
            event="engine_generate_setup",
            duration=setup_end_time - setup_start_time,
            extra={"request_id": request_id} if request_id else None,
            workid=self._rank,
            step=self.step
        )
        torch.cuda.synchronize()
        engine_call_start_time = time.time()
        output = await self._engine.async_generate(
            input_ids=generation_prompt_ids,
            sampling_params=kwargs,
            return_logprob=False,
            image_data=image_data,
        )
        torch.cuda.synchronize()
        engine_call_end_time = time.time()
        self.log_manager.log(
            log_path,
            event="engine_async_generate_actual",
            duration=engine_call_end_time - engine_call_start_time,
            extra={"request_id": request_id} if request_id else None,
            workid=self._rank,
            step=self.step
        )
        return output

    async def _handle_pending_state(self, _req: AsyncRolloutRequest) -> AsyncRolloutRequest:
        log_path = os.path.join(
            self.log_dir,
            f"step_{self.step}",
            f"worker_{self._rank}.jsonl"
        )
        if _req.tool_schemas is not None:
            torch.cuda.synchronize()
            tool_creation_start_time = time.time()
            tool_creation_coroutines = []
            for tool_schema in _req.tool_schemas:
                tool = self._tool_map[tool_schema.function.name]
                create_kwargs = _req.tools_kwargs[tool.name].get("create_kwargs", {})
                tool_creation_coroutines.append(tool.create(_req.request_id, **create_kwargs))
            tool_creation_results = await asyncio.gather(*tool_creation_coroutines)
            _req.add_tool_response_messages(
                self.processing_class, [tool_result for _, tool_result in tool_creation_results]
            )
            torch.cuda.synchronize()
            tool_creation_end_time = time.time()
            self.log_manager.log(
                log_path,
                event="tool_creation_pending_state",
                duration=tool_creation_end_time - tool_creation_start_time,
                extra={"request_id": _req.request_id},
                workid=self._rank,
                step=self.step
            )
        if _req.interaction_kwargs and self.interaction_map:
            torch.cuda.synchronize()
            interaction_start_time = time.time()
            interaction_kwargs = _req.interaction_kwargs
            # Get interaction by name from interaction_kwargs
            interaction_name = interaction_kwargs.get("name", "gsm8k")  # Default to gsm8k for backward compatibility
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )

            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(_req.request_id, **interaction_kwargs)
            torch.cuda.synchronize()
            interaction_end_time = time.time()
            self.log_manager.log(
                log_path,
                event="interaction_start_pending_state",
                duration=interaction_end_time - interaction_start_time,
                extra={"request_id": _req.request_id},
                workid=self._rank,
                step=self.step
            )

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _req_level_generate_sequences(self, prompts: DataProto, step: int = 0, **kwargs) -> DataProto:
        """Generates multi-turn sequences for a batch of prompts.
        For multi-turn generation, each prompt is processed separately via
        `_req_level_generate_sequences` for better tool calling control.
        Note that in multi-turn generation, we repeat the prompts for rollout.n times in ray_trainer.
        Thus we do not need to repeat the prompts here and set the sampling parameter n to 1.
        
        Args:
            prompts: DataProto containing input prompts
            step: Global training step (passed from generate_sequences)
            **kwargs: Additional arguments
        """
        # 保存 step 到 self.step，供后续方法使用
        self.step = step
        
        # 生成 log 文件名（每个 step 唯一）
        log_path = None
        if self._tp_rank == 0:
            log_path = os.path.join(
                self.log_dir,
                f"step_{step}",
                f"worker_{self._rank}.jsonl"
            )

        # Async rollout with tools support
        torch.cuda.synchronize()
        start_time = time.time()
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device

        # Initialize incremental Phase A evaluator reference (may be set later)
        _incremental_phase_a = None

        if self._tp_rank == 0:
            torch.cuda.synchronize()
            preprocess_start_time = time.time()
            req_list = self._preprocess_prompt_to_async_rollout_requests(
                prompts,
            )
            torch.cuda.synchronize()
            preprocess_end_time = time.time()
            self.log_manager.log(
                log_path,
                event="preprocessing_duration",
                duration=preprocess_end_time - preprocess_start_time,
                workid=self._rank,
                step=self.step
            )
            # distinguish training and validation
            if is_validate:
                # Validation mode: process all requests without abort
                loop = asyncio.get_event_loop()
                output_req_list = loop.run_until_complete(
                    asyncio.gather(
                        *[self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) for req in req_list],
                    )
                )
            else:
                # add progress monitoring and abort function
                total_requests = len(req_list)
                target_completion = int(total_requests * (1 - self.config.over_sample_rate))
                # abort when target_completion of requests are completed

                completed_count = 0
                aborted_requests = []
                all_tasks = []

                # === Incremental Phase A Setup ===
                # If Phase A config is provided in meta_info, create an incremental
                # evaluator that processes samples as they complete rollout.
                _incremental_phase_a = None
                _phase_a_extra_info_list = None
                _phase_a_config = prompts.meta_info.get("incremental_phase_a_config", None)
                if _phase_a_config and self._tp_rank == 0:
                    try:
                        from verl.utils.reward_score.process_reward_gig import IncrementalPhaseAEvaluator
                        _incremental_phase_a = IncrementalPhaseAEvaluator(
                            batch_size=total_requests,
                            model=_phase_a_config.get("model", "gpt-4o"),
                            num_workers=_phase_a_config.get("num_workers", 64),
                            timeout=_phase_a_config.get("timeout", 600.0),
                            api_provider=_phase_a_config.get("api_provider", "vendor_a"),
                            api_key=_phase_a_config.get("api_key"),
                            fallback_models=_phase_a_config.get("fallback_models"),
                            global_timeout=_phase_a_config.get("global_timeout", 600.0),
                        )
                        # Always get extra_info from prompts.non_tensor_batch (correctly chunked per DP rank),
                        # NOT from meta_info (which is shared across all DP ranks and would cause index mismatch).
                        _phase_a_extra_info_list = prompts.non_tensor_batch.get("extra_info", [{}] * total_requests)
                        print(
                            f"[IncrementalPhaseA] Enabled in rollout engine: "
                            f"batch_size={total_requests}, model={_phase_a_config.get('model', 'gpt-4o')}, "
                            f"extra_info_available={_phase_a_extra_info_list is not None and len(_phase_a_extra_info_list) > 0}"
                        )
                    except Exception as _e:
                        print(f"[IncrementalPhaseA] WARNING: Failed to initialize: {_e}")
                        _incremental_phase_a = None

                async def rollout_a_request_with_cancellation_handler(req):
                    try:
                        result = await self._async_rollout_a_request(req, do_sample, is_validate, **kwargs)
                        # print(f"[DEBUG] _async_rollout_a_request result: {result}")

                        # === Incremental Phase A: submit completed request ===
                        if _incremental_phase_a is not None and result.response_ids is not None:
                            try:
                                _sample_idx = result.batch_data_id
                                _resp_text = self.processing_class.decode(
                                    result.response_ids.squeeze(0), skip_special_tokens=True
                                )
                                _ei = {}
                                if _phase_a_extra_info_list is not None and _sample_idx < len(_phase_a_extra_info_list):
                                    _ei = _phase_a_extra_info_list[_sample_idx]
                                    # Ensure _ei is a proper Python dict
                                    if hasattr(_ei, 'item'):
                                        _ei = _ei.item()
                                    if not isinstance(_ei, dict):
                                        try:
                                            _ei = dict(_ei)
                                        except (TypeError, ValueError):
                                            _ei = {}
                                _q = _ei.get("question", "")
                                _cgt = _ei.get("constraints_gt", [])
                                _egt = _ei.get("entities_gt", {})
                                # Handle numpy arrays/objects from parquet serialization
                                if hasattr(_cgt, 'tolist'):
                                    _cgt = _cgt.tolist()
                                if not isinstance(_egt, dict):
                                    # Could be numpy void, structured array, etc.
                                    try:
                                        _egt = dict(_egt)
                                    except (TypeError, ValueError):
                                        _egt = {}
                                _rubrics = []
                                if _cgt and _egt:
                                    from verl.utils.reward_score.process_reward_gig import build_rubrics_from_constraints_gt
                                    _rubrics = build_rubrics_from_constraints_gt(_cgt, _egt)
                                # Debug: log first sample's rubrics info
                                if _sample_idx == 0:
                                    print(
                                        f"[IncrementalPhaseA] Sample 0 debug: "
                                        f"ei_type={type(_phase_a_extra_info_list[0]).__name__}, "
                                        f"ei_keys={list(_ei.keys()) if isinstance(_ei, dict) else 'N/A'}, "
                                        f"cgt_len={len(_cgt) if _cgt else 0}, "
                                        f"egt_len={len(_egt) if _egt else 0}, "
                                        f"rubrics_len={len(_rubrics)}"
                                    )
                                _incremental_phase_a.submit(
                                    sample_idx=_sample_idx,
                                    trajectory=_resp_text,
                                    question=_q,
                                    rubrics=_rubrics,
                                )
                            except Exception as _submit_err:
                                logger.warning(
                                    f"[IncrementalPhaseA] Failed to submit sample {result.batch_data_id}: {_submit_err}"
                                )
                                import traceback as _tb_submit
                                _tb_submit.print_exc()

                        return result
                    except asyncio.CancelledError:
                        # request is cancelled, return padding
                        logger.info(f"Request {req.request_id} was cancelled, creating padding")
                        aborted_requests.append(req.request_id)
                        return self._create_padding_request(req)

                async def run_with_cancellation():
                    nonlocal all_tasks
                    nonlocal completed_count
                    all_tasks = [
                        asyncio.create_task(rollout_a_request_with_cancellation_handler(req)) for req in req_list
                    ]

                    # Wait for target_completion tasks to complete
                    try:
                        for completed_task in asyncio.as_completed(all_tasks):
                            await completed_task
                            completed_count += 1
                            if completed_count >= target_completion:
                                break
                    finally:
                        # Cancel remaining tasks
                        for t in all_tasks:
                            if not t.done():
                                t.cancel()

                        # Wait for all tasks to finish (including cancelled ones)
                        final_results = await asyncio.gather(*all_tasks, return_exceptions=True)
                        # Abort all requests in SGLang engine
                        await self._engine.abort_request(abort_all=True)
                    return final_results

                torch.cuda.synchronize()
                async_generate_start_time = time.time()
                loop = asyncio.get_event_loop()
                output_req_list = loop.run_until_complete(run_with_cancellation())
                # print(f"[DEBUG]output_req_list: {output_req_list}")
                torch.cuda.synchronize()
                async_generate_end_time = time.time()
                self.log_manager.log(
                    log_path,
                    event="async_generate_duration",
                    duration=async_generate_end_time - async_generate_start_time,
                    workid=self._rank,
                    step=self.step
                )
            torch.cuda.synchronize()
            sort_start_time = time.time()
            sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))
            torch.cuda.synchronize()
            sort_end_time = time.time()
            self.log_manager.log(
                log_path,
                event="sorting_duration",
                duration=sort_end_time - sort_start_time,
                workid=self._rank,
                step=self.step
            )
        else:
            sorted_output_req_list = None

        torch.cuda.synchronize()
        barrier_start_time = time.time()
        dist.barrier()
        torch.cuda.synchronize()
        barrier_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="barrier_wait_duration",
                duration=barrier_end_time - barrier_start_time,
                workid=self._rank,
                step=self.step
            )

        torch.cuda.synchronize()
        broadcast_start_time = time.time()
        [sorted_output_req_list] = broadcast_pyobj(
            data=[sorted_output_req_list],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )
        torch.cuda.synchronize()
        broadcast_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="broadcast_duration",
                duration=broadcast_end_time - broadcast_start_time,
                workid=self._rank,
                step=self.step
            )
        # ========================================
        # === 第一步：从 AsyncRolloutRequest 提取数据 ===
        # ========================================
        # 
        # 在 _async_rollout_a_request 完成后，每个 req 已经被 finalize()，包含：
        # - prompt_ids/response_ids: token序列
        # - attention_mask/position_ids/loss_mask: 对应的mask和位置信息
        # - messages: 完整对话历史（包括system, user, assistant, tool消息）
        # - reward_scores: 各工具的奖励分数
        # - multi_modal_inputs: 多模态输入（如果有）
        # 
        # 现在需要从所有 req 中提取这些信息，构造成 batch 形式的 DataProto
        
        torch.cuda.synchronize()
        postprocess_start_time = time.time()

        # === 初始化列表用于收集数据 ===
        # 这些列表用于从每个 AsyncRolloutRequest 中收集数据
        prompt_ids, response_ids = [], []  # list[Tensor[seq_len]]: 每个req的prompt和response token IDs
        prompt_attention_mask, response_attention_mask = [], []  # list[Tensor[seq_len]]: attention masks
        prompt_position_ids, response_position_ids = [], []  # list[Tensor[seq_len]]: position IDs
        prompt_loss_mask, response_loss_mask = [], []  # list[Tensor[seq_len]]: loss masks（区分哪些token计算loss）
        messages = []  # list[dict]: 对话历史，每个元素为 {"messages": [...]}
        reward_scores = []  # list[dict]: 奖励分数，每个元素为 {"tool_name": [score1, score2, ...], ...}
        multi_modal_inputs = []  # list[dict]: 多模态输入（图像、视频等）
        request_ids = []  # list[str]: 请求ID
        if self.config.calculate_log_probs:
            output_logprobs = []  # list[Tensor]: log probabilities
            rollout_output_token_ids = []  # list[Tensor]: rollout时的output token IDs
        
        # ========================================
        # === 异常标记和指标收集列表 ===
        # ========================================
        # 用于收集每个请求的异常标记和统计指标
        abnormal_flags_list = []  # list[dict]: 每个请求的异常标记
        metrics_list = []  # list[dict]: 每个请求的统计指标

        # === 遍历所有完成的请求，提取数据 ===
        for req in sorted_output_req_list:
            # === 验证请求状态 ===
            # 每个 req 必须是 COMPLETED 状态，且所有序列长度一致
            assert req.state == AsyncRolloutRequestStateEnum.COMPLETED, f"Request {req.request_id} is not completed"
            assert (
                req.input_ids.shape[-1]
                == req.attention_mask.shape[-1]
                == req.position_ids.shape[-1]
                == req.loss_mask.shape[-1]
            ), f"""Request {req.request_id} has different length of
                {req.input_ids.shape[-1]=}, {req.attention_mask.shape[-1]=},
                {req.position_ids.shape[-1]=}, {req.loss_mask.shape[-1]=}"""
            error_message_lines = [
                f"""Request {req.request_id} has input_ids length {req.input_ids.shape[-1]}
                    greater than max_model_len {self.config.max_model_len}""",
                f"Decoded input_ids: {self.processing_class.decode(req.input_ids.squeeze(0))}",
                f"Decoded prompt_ids: {self.processing_class.decode(req.prompt_ids.squeeze(0))}",
                f"Decoded response_ids: {self.processing_class.decode(req.response_ids.squeeze(0))}",
                f"Messages: {req.messages}",
                f"Max model length: {req.max_model_len}",
            ]
            error_message = "\n".join(error_message_lines)
            assert req.input_ids.shape[-1] <= self.config.max_model_len, error_message

            # === 提取 Token 序列 ===
            # req.prompt_ids: [1, prompt_len] - 原始prompt（固定不变）
            # req.response_ids: [1, response_len] - 所有生成内容（assistant生成 + tool观察）
            #   在 finalize() 中计算：response_ids = input_ids[:, prompt_ids.shape[-1]:]
            # 
            # 示例（Multi-turn Tool Use）：
            # prompt_ids:   [sys, user, <im_start>assistant\n]  # 长度50
            # response_ids: [assistant(think1), tool_call1, tool(obs1), 
            #                assistant(think2), tool_call2, tool(obs2),
            #                assistant(final_answer)]  # 长度750
            # 
            # squeeze(0) 移除 batch 维度，从 [1, seq_len] 变为 [seq_len]
            prompt_ids.append(req.prompt_ids.to(tgt_device).squeeze(0))
            response_ids.append(req.response_ids.to(tgt_device).squeeze(0))


            # debug_message_lines = [
            #     f"[DEBUG]0input_ids: {req.input_ids[0].tolist()}",
            #     f"[DEBUG]0prompt_ids: {req.prompt_ids[0].tolist()}",
            #     f"[DEBUG]0response_ids: {req.response_ids[0].tolist()}",
            #     f"[DEBUG]0Decoded input_ids: {self.processing_class.decode(req.input_ids[0])}",
            #     f"[DEBUG]0Decoded prompt_ids: {self.processing_class.decode(req.prompt_ids[0])}",
            #     f"[DEBUG]0Decoded response_ids: {self.processing_class.decode(req.response_ids[0])}",
            # ]
            # debug_message = "\n".join(debug_message_lines)
            # logger.debug(f"logger[DEBUG]0debug_message: {debug_message}")
            # print(f"print[DEBUG]0debug_message: {debug_message}")


            # === 检查长度 ===
            # 检查 response_ids 是否超过 max_response_len
            # 如果超过，需要设置 exceed_max_tokens 标记，因为 response 会在 truncate_output_ids 中被截断
            if req.response_ids.shape[-1] > self.config.response_length:
                # 重要修复（2026-01-01）：
                # 之前只打印 warning，没有设置异常标记
                # 但 response_ids 在 truncate_output_ids 中已经被截断到 max_response_len
                # 这会导致保存的 rollout_data 中 output 不完整（例如缺少 <answer> 部分）
                # 
                # 修复：设置 exceed_max_tokens 标记，让 reward manager 知道这是异常样本
                req.exceed_max_tokens = True
                logger.warning(
                    f"""{req.request_id=} has response_ids length {req.response_ids.shape[-1]}
                    greater than max_response_len {self.config.response_length}, marking as exceed_max_tokens"""
                )
            
            # === 提取 Attention Masks ===
            # attention_mask: 标记哪些位置是真实token（1）、哪些是padding（0）
            # - prompt_attention_mask: [prompt_len] - prompt部分的mask
            # - response_attention_mask: [response_len] - response部分的mask
            prompt_attention_mask.append(req.prompt_attention_mask.to(tgt_device).squeeze(0))
            response_attention_mask.append(req.response_attention_mask.to(tgt_device).squeeze(0))
            
            # === 提取 Position IDs ===
            # position_ids: 标记每个token的位置编号（用于位置编码）
            # - prompt_position_ids: [prompt_len] - prompt部分的位置编号
            # - response_position_ids: [response_len] - response部分的位置编号
            prompt_position_ids.append(req.prompt_position_ids.to(tgt_device).squeeze(0))
            response_position_ids.append(req.response_position_ids.to(tgt_device).squeeze(0))
            
            # === 提取 Loss Masks ===
            # loss_mask: 标记哪些位置计算loss（1）、哪些位置不计算loss（0）
            # - prompt_loss_mask: [prompt_len] - 通常全为0（prompt不计算loss）
            # - response_loss_mask: [response_len] - assistant生成的token为1，tool观察的token为0
            # 
            # 示例（Multi-turn Tool Use）：
            # response_loss_mask: [1,1,1,...,1, 0,0,0,...,0, 1,1,1,...,1, 0,0,0,...,0, 1,1,1,...,1]
            #                      ^assistant1   ^tool1       ^assistant2   ^tool2       ^assistant3
            # 
            # 这样可以确保只对模型生成的部分计算loss，而不对工具返回的观察结果计算loss
            prompt_loss_mask.append(req.prompt_loss_mask.to(tgt_device).squeeze(0))
            response_loss_mask.append(req.response_loss_mask.to(tgt_device).squeeze(0))
            
            # === 提取非张量数据 ===
            # messages: list[Message] - 完整对话历史
            #   示例: [
            #     {"role": "system", "content": "..."},
            #     {"role": "user", "content": "..."},
            #     {"role": "assistant", "content": "...", "tool_calls": [...]},
            #     {"role": "tool", "content": "观察结果1"},
            #     {"role": "assistant", "content": "...", "tool_calls": [...]},
            #     {"role": "tool", "content": "观察结果2"},
            #     {"role": "assistant", "content": "最终答案"}
            #   ]
            messages.append({"messages": req.messages})
            
            # reward_scores: dict[str, list[float]] - 各工具的奖励分数
            #   示例: {"search_tool": [0.5, 0.3], "user_turn_rewards": [0.8]}
            reward_scores.append(req.reward_scores)
            
            # multi_modal_inputs: dict - 多模态输入（如果有）
            multi_modal_inputs.append(req.multi_modal_inputs)
            
            # request_id: str - 请求的唯一标识符
            request_ids.append(req.request_id)
            
            # === 提取 Log Probabilities（如果需要）===
            if self.config.calculate_log_probs:
                # output_logprobs: Tensor[response_len] - response部分每个token的log probability
                # rollout_output_token_ids: Tensor[response_len] - rollout时生成的token IDs
                # 
                # 注意：req.rollout_log_probs 是整个 input_ids 的 log_probs
                # 这里只提取 response 部分（最后 len(req.response_ids) 个）
                output_logprobs.append(req.rollout_log_probs[-len(req.response_ids) :])
                rollout_output_token_ids.append(req.output_token_ids[-len(req.response_ids) :])
            
            # ========================================
            # === 收集异常标记 ===
            # ========================================
            # 检测是否有未知工具调用（unknown_tool_names 不为空）
            has_unknown_tool = len(req.unknown_tool_names) > 0
            
            abnormal_flags = {
                "excessive_tool_calls_per_turn": req.excessive_tool_calls_per_turn,
                "tool_parse_error": req.tool_parse_error,
                "repeated_query": req.repeated_query,
                "search_error": req.search_error,
                "exceed_max_turns": req.exceed_max_turns,
                "exceed_max_tokens": req.exceed_max_tokens,
                "unknown_tool_in_trajectory": has_unknown_tool,  # 新增：未知工具调用标记
            }
            abnormal_flags_list.append(abnormal_flags)
            
            # ========================================
            # === 收集轨迹级别特征（用于单条轨迹reward计算）===
            # ========================================
            # 
            # 这些是每条轨迹的原始特征，不做聚合，直接保存到 DataProto
            # 用途：
            # - reward manager 可以根据这些特征为每条轨迹计算不同的reward
            # - 分析工具可以查看每条轨迹的详细信息
            # 
            trajectory_features = {
                # === 核心行为特征 ===
                "search_steps": req.total_assistant_turns,  # 搜索步骤数（有多少轮assistant消息）
                "subq_calls": req.total_tool_calls,  # 子查询调用总数（所有query_list的总长度）
                "tool_call_attempts": req.total_tool_call_attempts,  # <tool_call>标签出现次数
                
                # === 异常情况（布尔标记）===
                # 这些字段直接复制自 abnormal_flags，但在这里明确列出便于使用
                "excessive_tool_calls_per_turn": req.excessive_tool_calls_per_turn,
                "tool_parse_error": req.tool_parse_error,
                "repeated_query": req.repeated_query,
                "search_error": req.search_error,
                "exceed_max_turns": req.exceed_max_turns,
                "exceed_max_tokens": req.exceed_max_tokens,
                
                # === 详细统计（用于深度分析）===
                "max_tool_calls_in_single_turn": req.max_tool_calls_in_single_turn,
                "tool_calls_per_turn": req.tool_calls_per_turn,  # 每轮的query数量列表
                "query_history": req.query_history,  # 所有query的历史记录
                "unknown_tool_names": req.unknown_tool_names,  # 未知工具名称列表
                "finish_reason_type": str(req.finish_reason_type) if req.finish_reason_type else None,
            }
            metrics_list.append(trajectory_features)

        # debug_message_lines = [
        #     f"[DEBUG]1Decoded prompt_ids: {self.processing_class.decode(prompt_ids[0])}",
        #     f"[DEBUG]1Decoded response_ids: {self.processing_class.decode(response_ids[0])}",
        # ]
        # debug_message = "\n".join(debug_message_lines)
        # print(f"[DEBUG]1debug_message: {debug_message}")

        torch.cuda.synchronize()
        data_extraction_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="data_extraction_duration",
                duration=data_extraction_end_time - postprocess_start_time,
                workid=self._rank,
                step=self.step
            )

        # ========================================
        # === 第二步：Padding 对齐序列长度 ===
        # ========================================
        # 
        # 由于每个 req 的 prompt 和 response 长度可能不同，需要 padding 到统一长度
        # - prompt: left padding（在左侧填充）
        # - response: right padding（在右侧填充）
        # 
        # 为什么要这样padding？
        # 1. prompt用left padding：因为LLM从左到右生成，真实内容应该靠右
        # 2. response用right padding：因为生成从左开始，真实内容应该靠左
        
        torch.cuda.synchronize()
        padding_start_time = time.time()

        # === Padding Prompt IDs ===
        # 输入: list[Tensor[prompt_len_i]] - 每个req的prompt长度不同
        # 输出: Tensor[batch_size, max_prompt_len] - 对齐到最长prompt的长度
        # 
        # pad_sequence 会找到最长的序列，然后将其他序列padding到这个长度
        # padding_side="left" 表示在左侧填充 pad_token_id
        # 
        # 示例：
        # 输入: [[1,2,3], [4,5,6,7,8]]
        # 输出: [[pad,pad,1,2,3], [4,5,6,7,8]]
        prompt_ids = pad_sequence(
            prompt_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        # 如果padding后的长度还不到配置的prompt_length，继续padding到配置长度
        if prompt_ids.shape[-1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True)
        
        # === Padding Response IDs ===
        # 输入: list[Tensor[response_len_i]] - 每个req的response长度不同
        # 输出: Tensor[batch_size, max_response_len] - 对齐到最长response的长度
        # 
        # 默认 padding_side="right"，在右侧填充 pad_token_id
        # 
        # 示例：
        # 输入: [[10,11,12,13], [20,21]]
        # 输出: [[10,11,12,13], [20,21,pad,pad]]
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[-1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        # === Padding Attention Masks ===
        # attention_mask 用0填充（0表示padding位置，不参与attention计算）
        # 
        # 示例（prompt_attention_mask，left padding）：
        # 输入: [[1,1,1], [1,1,1,1,1]]
        # 输出: [[0,0,1,1,1], [1,1,1,1,1]]
        prompt_attention_mask = pad_sequence(
            prompt_attention_mask,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )
        if prompt_attention_mask.shape[-1] < self.config.prompt_length:
            prompt_attention_mask = pad_sequence_to_length(
                prompt_attention_mask, self.config.prompt_length, 0, left_pad=True
            )
        
        # 示例（response_attention_mask，right padding）：
        # 输入: [[1,1,1,1], [1,1]]
        # 输出: [[1,1,1,1], [1,1,0,0]]
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[-1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)

        # padding prompt_position_ids
        if prompt_position_ids[0].dim() == 2:
            # if prompt_position_ids is a 2D tensor
            # e.g. from qwen2vl, prompt_position_ids.shape = (3, seq_len)
            transposed_prompt_position_ids = [p.transpose(0, 1) for p in prompt_position_ids]
            prompt_position_ids = pad_sequence(
                transposed_prompt_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
            prompt_position_ids = prompt_position_ids.transpose(1, 2)
        else:
            prompt_position_ids = pad_sequence(
                prompt_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
        if prompt_position_ids.shape[-1] < self.config.prompt_length:
            prompt_position_ids = pad_sequence_to_length(
                prompt_position_ids, self.config.prompt_length, 0, left_pad=True
            )

        # padding response_position_ids
        if response_position_ids[0].dim() == 2:
            # if response_position_ids is a 2D tensor
            # e.g. from qwen2vl, response_position_ids.shape = (3, seq_len)
            transposed_response_position_ids = [p.transpose(0, 1) for p in response_position_ids]
            response_position_ids = pad_sequence(
                transposed_response_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
            response_position_ids = response_position_ids.transpose(1, 2)
        else:
            response_position_ids = pad_sequence(response_position_ids, batch_first=True, padding_value=0)
        if response_position_ids.shape[-1] < self.config.response_length:
            response_position_ids = pad_sequence_to_length(response_position_ids, self.config.response_length, 0)

        # === Padding Loss Masks ===
        # loss_mask 用0填充（0表示不计算loss的位置）
        # 
        # 关键点：loss_mask 决定了训练时哪些token参与loss计算
        # - prompt_loss_mask: 通常全为0（不对prompt计算loss）
        # - response_loss_mask: assistant生成的token为1，tool观察的token为0
        # 
        # 示例（response_loss_mask）：
        # 原始: [1,1,1,1, 0,0,0,0, 1,1,1,1]  # assistant1, tool1, assistant2
        #        ^计算loss ^不计算   ^计算loss
        # Padding后: [1,1,1,1, 0,0,0,0, 1,1,1,1, 0,0,0,0]  # 右侧填充0
        prompt_loss_mask = pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")
        if prompt_loss_mask.shape[1] < self.config.prompt_length:
            prompt_loss_mask = pad_sequence_to_length(prompt_loss_mask, self.config.prompt_length, 0, left_pad=True)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
        if self.config.calculate_log_probs:
            output_logprobs = pad_sequence(output_logprobs, padding_value=0.0, batch_first=True)
            output_logprobs = pad_sequence_to_length(
                output_logprobs, pad_token_id=0.0, max_seq_len=response_ids.shape[-1]
            ).to(tgt_device)
            rollout_output_token_ids = pad_sequence(
                rollout_output_token_ids, padding_value=self.pad_token_id, batch_first=True
            )
            rollout_output_token_ids = pad_sequence_to_length(
                rollout_output_token_ids, pad_token_id=self.pad_token_id, max_seq_len=response_ids.shape[-1]
            ).to(tgt_device)

        torch.cuda.synchronize()
        padding_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="padding_duration",
                duration=padding_end_time - padding_start_time,
                workid=self._rank,
                step=self.step
            )

        # ========================================
        # === 第三步：拼接 Prompt 和 Response ===
        # ========================================
        # 
        # 将 prompt 和 response 拼接成完整序列
        # 
        # 拼接后的结构：
        # input_ids: [batch_size, prompt_length + response_length]
        # = [pad, pad, ..., sys, user, gen_prompt, assistant1, tool1, assistant2, tool2, ..., pad, pad]
        #     ^左padding     ^prompt部分         ^response部分                                 ^右padding

        # 对应的 attention_mask:
        # = [0, 0, ..., 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ..., 0, 0]
        #     ^左padding    ^真实token                          ^右padding

        # 对应的 loss_mask（通过 response_mask 传递）:
        # = [0, 0, ..., 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, ..., 0, 0]
        #     ^prompt不计算loss        ^ass1  ^tool1 ^ass2     ^padding

        torch.cuda.synchronize()
        concatenation_start_time = time.time()

        # 在最后一个维度（序列维度）上拼接
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        torch.cuda.synchronize()
        concatenation_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="concatenation_duration",
                duration=concatenation_end_time - concatenation_start_time,
                workid=self._rank,
                step=self.step
            )
        # ========================================
        # === 第四步：构造 TensorDict (batch) ===
        # ========================================
        # 
        # 将所有张量数据打包成 TensorDict，这是 DataProto.batch 的内容
        # 
        # TensorDict 字段说明：
        # - prompts: [batch_size, prompt_length] - 原始prompt token IDs
        # - responses: [batch_size, response_length] - 生成的response token IDs
        # - response_mask: [batch_size, response_length] - response的loss mask
        #     * 值为1的位置：计算loss（assistant生成的token）
        #     * 值为0的位置：不计算loss（tool观察、padding）
        # - input_ids: [batch_size, prompt_length + response_length] - 完整序列
        # - attention_mask: [batch_size, prompt_length + response_length] - attention mask
        # - position_ids: [batch_size, prompt_length + response_length] - position IDs
        # 
        # 如果启用了 calculate_log_probs，还会包含：
        # - rollout_log_probs: [batch_size, response_length] - response的log probabilities
        # - rollout_output_token_ids: [batch_size, response_length] - rollout时的token IDs
        
        torch.cuda.synchronize()
        batch_construction_start_time = time.time()
        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [batch_size, prompt_length]
                "responses": response_ids,  # [batch_size, response_length]
                "response_mask": response_loss_mask,  # [batch_size, response_length] - 关键！区分哪些token计算loss
                "input_ids": input_ids,  # [batch_size, total_length] - 完整序列
                "attention_mask": attention_mask,  # [batch_size, total_length]
                "position_ids": position_ids,  # [batch_size, total_length]
            },
            batch_size=len(sorted_output_req_list),
        )

        # debug_message_lines = [
        #     f"[DEBUG]2input_ids: {input_ids[0].tolist()}",
        #     f"[DEBUG]2Decoded input_ids: {self.processing_class.decode(input_ids[0])}",
        #     f"[DEBUG]2Decoded prompt_ids: {self.processing_class.decode(prompt_ids[0])}",
        #     f"[DEBUG]2Decoded response_ids: {self.processing_class.decode(response_ids[0])}",
        # ]
        # debug_message = "\n".join(debug_message_lines)
        # print(f"[DEBUG]2debug_message: {debug_message}")

        torch.cuda.synchronize()
        batch_construction_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="batch_construction_duration",
                duration=batch_construction_end_time - batch_construction_start_time,
                workid=self._rank,
                step=self.step
            )
        if self.config.calculate_log_probs:
            batch["rollout_log_probs"] = output_logprobs
            batch["rollout_output_token_ids"] = rollout_output_token_ids

        # free cache engine
        torch.cuda.synchronize()
        cache_flush_start_time = time.time()
        if self._engine is not None and self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())
        torch.cuda.synchronize()
        cache_flush_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="cache_flush_duration",
                duration=cache_flush_end_time - cache_flush_start_time,
                workid=self._rank,
                step=self.step
            )

        # ========================================
        # === 第五步：构造 non_tensor_batch ===
        # ========================================
        # 
        # non_tensor_batch 包含无法转换为张量的数据（字典、列表、字符串等）
        # 这些数据对应 DataProto.non_tensor_batch
        # 
        # non_tensor_batch 字段说明：
        # - messages: np.array[dict] - 每个样本的对话历史
        #     示例: [{"messages": [
        #              {"role": "system", "content": "..."},
        #              {"role": "user", "content": "..."},
        #              {"role": "assistant", "content": "...", "tool_calls": [...]},
        #              {"role": "tool", "content": "观察结果1"},
        #              ...
        #            ]}]
        # 
        # - reward_scores: np.array[dict] - 每个样本的奖励分数
        #     示例: [{"search_tool": [0.5, 0.3], "user_turn_rewards": [0.8]}]
        #     这些分数用于后续的reward计算和策略优化
        # 
        # - request_id: np.array[str] - 每个样本的请求ID
        #     用于追踪和调试
        # 
        # - multi_modal_inputs (如果有): np.array[dict] - 多模态输入
        #     包含图像、视频等非文本数据
        
        torch.cuda.synchronize()
        final_construction_start_time = time.time()
        
        # ========================================
        # === 计算批级别的聚合指标（用于监控和分析）===
        # ========================================
        # 
        # 这些是批次级别的聚合统计，只计算占比、平均值、中位数等
        # 用途：
        # - 监控模型训练过程中的行为趋势
        # - 发现异常批次并及时调整
        # 
        batch_size = len(abnormal_flags_list)
        
        # === 1. 统计各类异常轨迹占比 ===
        # 
        # 异常分类：
        # - repeated_query: 重复查询（模型陷入循环）
        # - tool_parse_error: 工具解析失败（JSON格式错误等）
        # - exceed_max_turns: 达到max_steps上限（工具调用次数>100）
        # - exceed_max_tokens: 超出token上限被truncate
        # - excessive_tool_calls_per_turn: 单轮大量subq调用
        # - search_error: 搜索错误（环境问题，如超时）
        # 
        repeated_query_count = sum(1 for f in abnormal_flags_list if f["repeated_query"])
        tool_parse_error_count = sum(1 for f in abnormal_flags_list if f["tool_parse_error"])
        exceed_max_turns_count = sum(1 for f in abnormal_flags_list if f["exceed_max_turns"])
        exceed_max_tokens_count = sum(1 for f in abnormal_flags_list if f["exceed_max_tokens"])
        excessive_tool_calls_count = sum(1 for f in abnormal_flags_list if f["excessive_tool_calls_per_turn"])
        search_error_count = sum(1 for f in abnormal_flags_list if f["search_error"])
        
        # 统计有未知工具的轨迹数量（使用 abnormal_flags 中的 unknown_tool_in_trajectory 字段）
        unknown_tool_count = sum(1 for f in abnormal_flags_list if f.get("unknown_tool_in_trajectory", False))
        
        # 统计正常样本（没有任何异常标记，包括 unknown_tool_in_trajectory）
        normal_count = sum(
            1 for f in abnormal_flags_list
            if not any([
                f["excessive_tool_calls_per_turn"],
                f["tool_parse_error"],
                f["repeated_query"],
                f["search_error"],
                f["exceed_max_turns"],
                f["exceed_max_tokens"],
                f.get("unknown_tool_in_trajectory", False),  # 新增：未知工具也算异常
            ])
        )
        
        # === 2. 统计search steps的分布 ===
        all_search_steps = [m["search_steps"] for m in metrics_list]
        
        # === 3. 统计subq调用数的分布 ===
        all_subq_calls = [m["subq_calls"] for m in metrics_list]
        
        # 统计所有未知工具名称（去重）
        all_unknown_tool_names = set()
        for m in metrics_list:
            all_unknown_tool_names.update(m["unknown_tool_names"])
        
        # 批次统计汇总（仅包含聚合指标）
        batch_statistics = {
            # === 批次基本信息 ===
            "batch_size": batch_size,
            "normal_count": normal_count,
            "normal_rate": round(normal_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            # === 异常轨迹占比 ===
            "repeated_query_count": repeated_query_count,
            "repeated_query_rate": round(repeated_query_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            "tool_parse_error_count": tool_parse_error_count,
            "tool_parse_error_rate": round(tool_parse_error_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            "exceed_max_turns_count": exceed_max_turns_count,
            "exceed_max_turns_rate": round(exceed_max_turns_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            "exceed_max_tokens_count": exceed_max_tokens_count,
            "exceed_max_tokens_rate": round(exceed_max_tokens_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            "excessive_tool_calls_count": excessive_tool_calls_count,
            "excessive_tool_calls_rate": round(excessive_tool_calls_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            "search_error_count": search_error_count,
            "search_error_rate": round(search_error_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            "unknown_tool_count": unknown_tool_count,
            "unknown_tool_rate": round(unknown_tool_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            # === search steps统计（平均、最大、中位数）===
            "search_steps_avg": round(np.mean(all_search_steps), 2) if all_search_steps else 0.0,
            "search_steps_median": round(np.median(all_search_steps), 2) if all_search_steps else 0.0,
            "search_steps_max": int(np.max(all_search_steps)) if all_search_steps else 0,
            "search_steps_min": int(np.min(all_search_steps)) if all_search_steps else 0,
            
            # === subq调用数统计（平均、最大、中位数）===
            "subq_calls_avg": round(np.mean(all_subq_calls), 2) if all_subq_calls else 0.0,
            "subq_calls_median": round(np.median(all_subq_calls), 2) if all_subq_calls else 0.0,
            "subq_calls_max": int(np.max(all_subq_calls)) if all_subq_calls else 0,
            "subq_calls_min": int(np.min(all_subq_calls)) if all_subq_calls else 0,
            
            # === 未知工具名称列表（去重后的完整列表）===
            "unique_unknown_tool_names": list(all_unknown_tool_names) if all_unknown_tool_names else [],
            "unique_unknown_tool_count": len(all_unknown_tool_names),
        }
        
        # 记录批次统计到日志
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="batch_statistics",
                extra=batch_statistics,
                workid=self._rank,
                step=self.step
            )
            
            # 同时打印到控制台（方便监控）
            logger.info(
                f"[Batch Statistics] Step {self.step}: "
                f"Size={batch_size}, Normal={normal_count}({batch_statistics['normal_rate']:.2%}), "
                f"Repeated={repeated_query_count}({batch_statistics['repeated_query_rate']:.2%}), "
                f"ParseErr={tool_parse_error_count}({batch_statistics['tool_parse_error_rate']:.2%}), "
                f"MaxTurns={exceed_max_turns_count}({batch_statistics['exceed_max_turns_rate']:.2%}), "
                f"MaxTokens={exceed_max_tokens_count}({batch_statistics['exceed_max_tokens_rate']:.2%}), "
                f"ExcessiveCalls={excessive_tool_calls_count}({batch_statistics['excessive_tool_calls_rate']:.2%})"
            )
            
            logger.info(
                f"[Batch Metrics] Step {self.step}: "
                f"SearchSteps(avg={batch_statistics['search_steps_avg']:.2f}, "
                f"median={batch_statistics['search_steps_median']:.2f}, "
                f"max={batch_statistics['search_steps_max']}), "
                f"SubqCalls(avg={batch_statistics['subq_calls_avg']:.2f}, "
                f"median={batch_statistics['subq_calls_median']:.2f}, "
                f"max={batch_statistics['subq_calls_max']})"
            )
            
            # 如果有未知工具，单独打印警告
            if all_unknown_tool_names:
                logger.warning(
                    f"[Unknown Tools] Step {self.step}: "
                    f"Found in {unknown_tool_count} trajectories ({batch_statistics['unknown_tool_rate']:.2%}), "
                    f"Unique tools: {list(all_unknown_tool_names)}"
                )
        
        # ========================================
        # === 构造 extra_info（兼容现有DataProto结构）===
        # ========================================
        # 
        # extra_info 包含每个样本的额外信息，用于后续reward计算和分析
        # 结构：extra_info = [sample1_info, sample2_info, ...]
        # 
        # 每个样本的 extra_info 包含：
        # - abnormal_flags: dict - 异常标记（布尔值）
        # - trajectory_metrics: dict - 轨迹特征（search_steps, subq_calls等）
        # 
        extra_info_list = []
        for i in range(len(abnormal_flags_list)):
            sample_extra_info = {
                # === 异常标记（用于判断是否给0 reward）===
                "abnormal_flags": abnormal_flags_list[i],
                
                # === 轨迹特征（用于单条轨迹reward计算）===
                "trajectory_metrics": metrics_list[i],
            }
            extra_info_list.append(sample_extra_info)
        
        non_tensor_batch = {
            "messages": np.array(messages, dtype=object),  # 对话历史
            "reward_scores": np.array(reward_scores, dtype=object),  # 奖励分数
            "request_id": np.array(request_ids, dtype=object),  # 请求ID
            
            # === rollout_extra_info: 每个样本的额外信息（rollout阶段生成）===
            # 改名为 rollout_extra_info 以避免与原始数据的 extra_info 冲突
            "rollout_extra_info": np.array(extra_info_list, dtype=object),
            
            # === batch_statistics: 批次级别的聚合统计（用于监控）===
            # 注意：batch_statistics 是一个dict，需要包装成包含batch_size个相同元素的array
            # 这样才能满足 DataProto.check_consistency() 的要求（所有非张量数据的第一维必须等于batch_size）
            "batch_statistics": np.array([batch_statistics] * batch_size, dtype=object),
        }

        is_multimodal = isinstance(self.processing_class, ProcessorMixin) and (
            hasattr(self.processing_class, "image_processor") or hasattr(self.model_hf_config, "vision_config")
        )

        if is_multimodal:
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs, dtype=object)

        # ========================================
        # === 第六步：构造最终的 DataProto ===
        # ========================================
        # 
        # DataProto 是整个 rollout 的输出格式，包含两部分：
        # 1. batch (TensorDict): 所有张量数据
        # 2. non_tensor_batch (dict): 所有非张量数据
        # 
        # DataProto 的完整结构：
        # {
        #   "batch": {
        #     "prompts": Tensor[batch_size, prompt_length],
        #     "responses": Tensor[batch_size, response_length],
        #     "response_mask": Tensor[batch_size, response_length],  # 区分计算loss的位置
        #     "input_ids": Tensor[batch_size, total_length],
        #     "attention_mask": Tensor[batch_size, total_length],
        #     "position_ids": Tensor[batch_size, total_length],
        #     "rollout_log_probs": Tensor[batch_size, response_length],  # 如果有
        #     "rollout_output_token_ids": Tensor[batch_size, response_length],  # 如果有
        #   },
        #   "non_tensor_batch": {
        #     "messages": np.array[dict],  # 对话历史
        #     "reward_scores": np.array[dict],  # 奖励分数
        #     "request_id": np.array[str],  # 请求ID
        #     "multi_modal_inputs": np.array[dict],  # 多模态输入（如果有）
        #   },
        #   "meta_info": {}  # 元信息（由调用方添加）
        # }
        # 
        # 这个 DataProto 会被传递给后续的训练流程（critic、actor等）
        result = DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
        )
        torch.cuda.synchronize()
        final_construction_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="final_construction_duration",
                duration=final_construction_end_time - final_construction_start_time,
                workid=self._rank,
                step=self.step
            )

        torch.cuda.synchronize()
        total_end_time = time.time()
        if self._tp_rank == 0:
            all_response_lengths = [len(req.response_ids.squeeze(0)) for req in sorted_output_req_list if req.response_ids is not None]
            all_actual_response_tokens = [torch.sum(req.response_loss_mask.squeeze(0)).item() for req in sorted_output_req_list if req.response_loss_mask is not None]


            self.log_manager.log(
                log_path,
                event="step_response_length_stats",
                extra={
                    "response_lengths": all_response_lengths,
                    "actual_response_tokens": all_actual_response_tokens,
                    "response_length_mean": np.mean(all_response_lengths),
                    "response_length_median": np.median(all_response_lengths),
                    "response_length_p80": np.percentile(all_response_lengths, 80),
                    "response_length_p95": np.percentile(all_response_lengths, 95),
                    "response_length_max": max(all_response_lengths),
                    "response_length_min": min(all_response_lengths),
                    "requests_over_500": sum(1 for x in all_response_lengths if x > 500),
                    "requests_over_1000": sum(1 for x in all_response_lengths if x > 1000),
                    "batch_size": len(sorted_output_req_list)
                },
                workid=self._rank,
                step=self.step
            )
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="total_step_duration",
                duration=total_end_time - start_time,
                workid=self._rank,
                step=self.step
            )

        # === Collect Incremental Phase A results ===
        # Phase A evaluator only runs on tp_rank==0 (master rank).
        # We need to broadcast the result to all TP ranks so that
        # the returned DataProto is consistent across ranks.
        #
        # IMPORTANT: Phase A results must be stored in non_tensor_batch (NOT meta_info)
        # because DataProto.concat() only keeps the first rank's meta_info,
        # discarding results from other DP ranks. non_tensor_batch is properly
        # concatenated via np.concatenate across all DP ranks.
        _phase_a_collected = None
        if _incremental_phase_a is not None:
            try:
                print(
                    f"[IncrementalPhaseA] Collecting results "
                    f"(submitted={_incremental_phase_a._submitted_count}/{_incremental_phase_a.batch_size})..."
                )
                _phase_a_collected = _incremental_phase_a.collect()
                print(
                    f"[IncrementalPhaseA] Results collected successfully: "
                    f"batch_size={_phase_a_collected['batch_size']}"
                )
            except Exception as _collect_err:
                print(f"[IncrementalPhaseA] WARNING: Failed to collect results: {_collect_err}")
                import traceback as _tb
                _tb.print_exc()

        # Broadcast Phase A result from master rank to all TP ranks
        # Only do this if Phase A config was provided (to avoid unnecessary
        # broadcast in validate mode or when Phase A is disabled).
        _phase_a_config_present = prompts.meta_info.get("incremental_phase_a_config", None) is not None
        if _phase_a_config_present:
            [_phase_a_collected] = broadcast_pyobj(
                data=[_phase_a_collected],
                rank=self._rank,
                dist_group=self._device_mesh_cpu["tp"].get_group(),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
        if _phase_a_collected is not None:
            # Store Phase A results as per-sample arrays in non_tensor_batch
            # so that DataProto.concat() correctly merges all DP ranks.
            _pa_batch_size = _phase_a_collected["batch_size"]
            result.non_tensor_batch["_phase_a_steps"] = np.array(
                _phase_a_collected["all_steps_per_sample"], dtype=object
            )
            result.non_tensor_batch["_phase_a_state_vectors"] = np.array(
                _phase_a_collected["corrected_state_vectors_per_sample"], dtype=object
            )
            # Mark that Phase A results are available (1 per sample for concat compatibility)
            result.non_tensor_batch["_phase_a_available"] = np.ones(_pa_batch_size, dtype=np.int32)
            print(
                f"[IncrementalPhaseA] Stored Phase A results in non_tensor_batch: "
                f"batch_size={_pa_batch_size}, "
                f"steps_shape={result.non_tensor_batch['_phase_a_steps'].shape}, "
                f"sv_shape={result.non_tensor_batch['_phase_a_state_vectors'].shape}"
            )

        return result

    def _create_padding_request(self, original_req: AsyncRolloutRequest) -> AsyncRolloutRequest:
        # create a padding request to replace the aborted request
        # the padding request has the following characteristics:
        # 1. state is COMPLETED, but contains empty response
        # 2. response_loss_mask is all 0, ensuring it is ignored in loss calculation
        # 3. keep the original request structure, but the content is empty
        # create padding response_ids (all pad_token_id)
        padding_response_length = self.config.response_length
        padding_response_ids = torch.full(
            (1, padding_response_length),
            self.pad_token_id,
            dtype=torch.long,
            device=original_req.input_ids.device if original_req.input_ids is not None else "cpu",
        )

        # create padding attention_mask (all 0)
        padding_response_attention_mask = torch.zeros(
            (1, padding_response_length),
            dtype=torch.long,
            device=original_req.attention_mask.device if original_req.attention_mask is not None else "cpu",
        )

        # create padding position_ids
        if original_req.position_ids is not None:
            prompt_length = original_req.prompt_ids.shape[-1] if original_req.prompt_ids is not None else 0
            padding_response_position_ids = torch.arange(
                prompt_length, prompt_length + padding_response_length, dtype=torch.long
            ).unsqueeze(0)
            if original_req.position_ids.dim() == 2:
                # if it is a 2D tensor (e.g. qwen2vl)
                padding_response_position_ids = padding_response_position_ids.repeat(
                    original_req.position_ids.shape[0], 1
                )
        else:
            padding_response_position_ids = None

        # create padding loss_mask (all 0, ensuring it is ignored)
        padding_response_loss_mask = torch.zeros(
            (1, padding_response_length),
            dtype=torch.long,
            device=original_req.loss_mask.device if original_req.loss_mask is not None else "cpu",
        )

        padding_req = AsyncRolloutRequest(
            batch_data_id=original_req.batch_data_id,
            rollout_offset=original_req.rollout_offset,
            request_id=original_req.request_id,
            state=AsyncRolloutRequestStateEnum.COMPLETED,
            messages=original_req.messages,
            multi_modal_keys=original_req.multi_modal_keys,
            multi_modal_data=original_req.multi_modal_data,
            multi_modal_inputs=original_req.multi_modal_inputs,
            tool_schemas=original_req.tool_schemas,
            tools_kwargs=original_req.tools_kwargs,
            interaction_kwargs=original_req.interaction_kwargs,
            input_ids=original_req.input_ids,
            prompt_ids=original_req.prompt_ids,
            response_ids=padding_response_ids,
            attention_mask=original_req.attention_mask,
            prompt_attention_mask=original_req.prompt_attention_mask,
            response_attention_mask=padding_response_attention_mask,
            position_ids=original_req.position_ids,
            prompt_position_ids=original_req.prompt_position_ids,
            response_position_ids=padding_response_position_ids,
            loss_mask=original_req.loss_mask,
            prompt_loss_mask=original_req.prompt_loss_mask,
            response_loss_mask=padding_response_loss_mask,
            reward_scores={},
            max_prompt_len=original_req.max_prompt_len,
            max_response_len=original_req.max_response_len,
            metrics={},
            output_token_ids=None,
            rollout_log_probs=None,
            use_inference_chat_template=original_req.use_inference_chat_template,
            tokenization_sanity_check_mode=original_req.tokenization_sanity_check_mode,
            generation_prompt_ids=original_req.generation_prompt_ids,
            base_conv_wo_gen_prompt_end_pos=original_req.base_conv_wo_gen_prompt_end_pos,
            base_conv_with_gen_prompt_end_pos=original_req.base_conv_with_gen_prompt_end_pos,
            processing_class=self.processing_class,
        )
        return padding_req

    def _preprocess_prompt_to_async_rollout_requests(self, prompts: DataProto, n: int = 1) -> list[AsyncRolloutRequest]:
        assert "raw_prompt" in prompts.non_tensor_batch, (
            "need data.return_raw_chat=True, due to no official way do parse_messages"
        )
        logger.info(
            "n is deprecated for SGLang rollout since ray ppo trainer will repeat the prompts for rollout.n times"
        )
        req_list = []
        multi_modal_data_list = prompts.non_tensor_batch.get(
            "multi_modal_data", [None] * len(prompts.non_tensor_batch["raw_prompt"])
        )

        for data_idx, (raw_prompt, multi_modal_data) in enumerate(
            zip(prompts.non_tensor_batch["raw_prompt"], multi_modal_data_list, strict=True)
        ):
            if self._tool_schemas:
                _tools_kwargs = prompts.non_tensor_batch["tools_kwargs"][data_idx]
                _tool_schemas = [self._tool_map[k].get_openai_tool_schema() for k in _tools_kwargs.keys()]
                _input_ids = None
                _attention_mask = None
            else:
                _input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][data_idx])
                _attention_mask = _pre_process_inputs(0, prompts.batch["attention_mask"][data_idx])
                _tools_kwargs = {}
                _tool_schemas = None

            if self.interaction_map:
                _interaction_kwargs = prompts.non_tensor_batch["interaction_kwargs"][data_idx]
            else:
                _interaction_kwargs = {}

            if not isinstance(raw_prompt, list | np.ndarray):
                raise TypeError(f"raw_prompt must be a list or numpy array, got {type(raw_prompt)}")

            req = AsyncRolloutRequest(
                batch_data_id=data_idx,
                rollout_offset=0,
                request_id=str(uuid4()),
                state=AsyncRolloutRequestStateEnum.PENDING,
                messages=list(raw_prompt),
                multi_modal_data=multi_modal_data,
                tool_schemas=_tool_schemas,
                tools_kwargs=_tools_kwargs,
                interaction_kwargs=_interaction_kwargs,
                input_ids=_input_ids,
                response_ids=None,
                attention_mask=_attention_mask,
                response_attention_mask=None,
                response_position_ids=None,
                response_loss_mask=None,
                reward_scores={},
                max_prompt_len=self.config.prompt_length,
                max_response_len=self.config.response_length,
                max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
                use_inference_chat_template=self.config.multi_turn.use_inference_chat_template,
                tokenization_sanity_check_mode=self.config.multi_turn.tokenization_sanity_check_mode,
                processing_class=self.processing_class,
            )
            error_message = f"""Request {req.request_id} has mismatched lengths:
            input_ids={req.input_ids.shape[-1]},
            attention_mask={req.attention_mask.shape[-1]},
            position_ids={req.position_ids.shape[-1]},
            loss_mask={req.loss_mask.shape[-1]}"""
            assert (
                req.input_ids.shape[-1]
                == req.attention_mask.shape[-1]
                == req.position_ids.shape[-1]
                == req.loss_mask.shape[-1]
            ), error_message
            req_list.append(req)

        return req_list

    # ==================== server mode public methods ====================

    async def chat_completion(self, json_request):
        """OpenAI chat completion API."""
        assert self._tp_rank == 0, "only called in tp rank 0"
        _input_ids = None
        _attention_mask = None
        _position_ids = None
        _tool_schemas = []
        _tools_kwargs = {}

        req = AsyncRolloutRequest(
            request_id=str(uuid4()),
            state=AsyncRolloutRequestStateEnum.PENDING,
            messages=[Message.model_validate(msg) for msg in json_request["messages"]],
            tool_schemas=_tool_schemas,
            tools_kwargs=_tools_kwargs,
            input_ids=_input_ids,
            prompt_ids=_input_ids,
            response_ids=None,
            attention_mask=_attention_mask,
            prompt_attention_mask=_attention_mask,
            response_attention_mask=None,
            position_ids=_position_ids,
            prompt_position_ids=_position_ids,
            response_position_ids=None,
            loss_mask=None,
            prompt_loss_mask=None,
            response_loss_mask=None,
            reward_scores={},
            max_prompt_len=self.config.prompt_length,
            max_response_len=self.config.response_length,
            max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
            use_inference_chat_template=self.config.multi_turn.use_inference_chat_template,
            tokenization_sanity_check_mode=self.config.multi_turn.tokenization_sanity_check_mode,
            processing_class=self.processing_class,
        )

        # json_request already contains sampling_params
        # Filter only valid SamplingParams arguments
        valid_sampling_params = {}
        temp_sampling_params = SamplingParams()  # Create temporary instance to check valid attributes
        for k, v in json_request.items():
            if k not in ["messages", "model", "tools"] and hasattr(temp_sampling_params, k):
                valid_sampling_params[k] = v
        output = await self._handle_engine_call(req, valid_sampling_params)
        # it can be Dict or AsyncIterator[Dict]
        if isinstance(output, dict):
            outputs = [output]
        else:
            outputs = output

        # build openai chat completion format
        choices = []
        id = None
        for i, content in enumerate(outputs):
            choices.append(
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": content["text"],
                    },
                    "finish_reason": content["meta_info"]["finish_reason"]["type"],
                }
            )
            id = content["meta_info"]["id"]

        return {
            "id": "chatcmpl-" + id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": json_request.get("model", "sglang_model"),
            "choices": choices,
        }

        # this function is left for uniform train-inference resharding

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> torch.Tensor:
        """Generate sequence with token-in-token-out."""
        request_sampling_params = self.sampling_params.copy()
        request_sampling_params.update(sampling_params)
        output = await self._handle_engine_generate(prompt_ids, request_sampling_params, image_data=image_data)
        return output["output_ids"]

    async def wake_up(self):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        await self.sharding_manager.wake_up()  # pylint: disable=C2801
        self.is_sleep = False

    async def sleep(self):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        await self.sharding_manager.sleep()
        self.is_sleep = True
