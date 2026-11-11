#!/usr/bin/env bash
# KnowGPO method only; external data, model and services are supplied by the caller.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=false
if [[ "${1:-}" == '--dry-run' ]]; then DRY_RUN=true; shift; fi
if [[ "${1:-}" == '--help' ]]; then
  printf '%s\n' 'Set KNOWGPO_MODEL_PATH, KNOWGPO_TRAIN_FILES, KNOWGPO_VAL_FILES, KNOWGPO_SEARCH_URL, KNOWGPO_JUDGE_API_KEY, KNOWGPO_JUDGE_BASE_URL, KNOWGPO_JUDGE_MODEL, then run this script [--dry-run] [Hydra overrides...]'; exit 0
fi
if [[ "$DRY_RUN" == false ]]; then
  for key in KNOWGPO_MODEL_PATH KNOWGPO_TRAIN_FILES KNOWGPO_VAL_FILES KNOWGPO_SEARCH_URL KNOWGPO_JUDGE_API_KEY KNOWGPO_JUDGE_BASE_URL KNOWGPO_JUDGE_MODEL; do
    if [[ -z "${!key:-}" ]]; then printf 'Missing required setting: %s\n' "$key" >&2; exit 2; fi
  done
fi
# Check defaults without accessing GPUs, files, network or the model.
cmd=(python -m verl.trainer.main_ppo
  "actor_rollout_ref.model.path=${KNOWGPO_MODEL_PATH:-/path/to/sft-model}"
  "data.train_files=${KNOWGPO_TRAIN_FILES:-/path/to/train.parquet}"
  "data.val_files=${KNOWGPO_VAL_FILES:-/path/to/val.parquet}"
  'data.train_batch_size=40'
  'data.return_raw_chat=True'
  'data.max_prompt_length=32768'
  'data.max_response_length=97280'
  'actor_rollout_ref.rollout.max_model_len=131072'
  'actor_rollout_ref.rollout.tensor_model_parallel_size=8'
  'actor_rollout_ref.rollout.gpu_memory_utilization=0.6'
  'actor_rollout_ref.rollout.max_num_batched_tokens=131072'
  'actor_rollout_ref.rollout.multi_turn.max_tool_response_length=14000'
  'actor_rollout_ref.actor.ppo_mini_batch_size=8'
  'actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576'
  'actor_rollout_ref.rollout.over_sample_rate=0.2'
  'actor_rollout_ref.rollout.name=sglang'
  'actor_rollout_ref.rollout.n=16'
  'actor_rollout_ref.rollout.temperature=0.6'
  'actor_rollout_ref.rollout.top_p=0.95'
  'actor_rollout_ref.rollout.top_k=20'
  'actor_rollout_ref.rollout.multi_turn.enable=True'
  'actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20'
  'actor_rollout_ref.rollout.multi_turn.max_user_turns=20'
  'actor_rollout_ref.rollout.multi_turn.format=qwen'
  'actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True'
  'actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable'
  "actor_rollout_ref.rollout.multi_turn.tool_config_path=$ROOT/verl/tools/search_tool_config.yaml"
  'algorithm.adv_estimator=grpo'
  'actor_rollout_ref.actor.use_kl_loss=True'
  'actor_rollout_ref.actor.kl_loss_coef=0.001'
  'algorithm.use_kl_in_reward=False'
  'reward_model.reward_manager=apiprimedapotrajectoryv7rgcir'
  'trainer.total_epochs=3'
  'trainer.logger=[console]'
  'trainer.project_name=knowgpo'
  'trainer.experiment_name=knowgpo'
  'trainer.val_before_train=False'
)
# Merge Hydra overrides by key rather than emitting duplicates (Hydra rejects
# duplicate keys in some versions). Keep method selection locked to KnowGPO.
for override in "$@"; do
  key="${override%%=*}"
  if [[ "$key" == 'reward_model.reward_manager' ]]; then
    printf '%s\n' 'Only the KnowGPO reward manager is supported by this launcher' >&2
    exit 2
  fi
  replaced=false
  for i in "${!cmd[@]}"; do
    if [[ "${cmd[i]}" == "$key="* ]]; then
      cmd[i]="$override"
      replaced=true
      break
    fi
  done
  if [[ "$replaced" == false ]]; then cmd+=("$override"); fi
done
if [[ "$DRY_RUN" == true ]]; then
  printf 'Working directory: %s\n' "$ROOT"
  printf 'Command: '; printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi
cd "$ROOT"
exec "${cmd[@]}"
