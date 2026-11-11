"""Private service defaults must not be shipped with the source tree."""
import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]


class TransportRedaction(unittest.TestCase):
    def test_google_helper_cookie_has_no_bundled_value(self):
        tree = ast.parse((ROOT / 'verl/tools/utils/call_as_xhub_toolcall_google.py').read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == 'Cookie':
                    self.assertFalse(isinstance(value, ast.Constant) and bool(value.value))

    def test_no_internal_ip_or_service_is_bundled(self):
        candidates = (
            'verl/tools/utils/call_visit.py',
            'verl/tools/utils/call_refiner_agent.py',
            'verl/tools/utils/call_xhub_get_qintent.py',
            'verl/tools/utils/call_xhub_get_qsite.py',
            'verl/tools/utils/call_xhub_rewrite_query.py',
            'verl/utils/reward_score/taiji_api_chat_completion.py',
        )
        pattern = re.compile(r'http://\d{1,3}(?:\.\d{1,3}){3}|\.production\.polaris')
        for relative in candidates:
            with self.subTest(file=relative):
                self.assertNotRegex((ROOT / relative).read_text(), pattern)

    def test_rollout_debug_path_is_not_private(self):
        text = (ROOT / 'verl/workers/rollout/sglang_rollout/sglang_rollout.py').read_text()
        self.assertNotRegex(text, r'_debug_log_dir\s*=\s*["\x27]/root/')
