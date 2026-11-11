import pathlib
import re
import subprocess
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[2]

class MethodConfig(unittest.TestCase):
    def test_v3_v4_judge_providers_are_client_supported(self):
        for suffix in ('v3_gig','v4_rgc'):
            text=(ROOT/f'verl/trainer/config/reward_dapo_trajectory_{suffix}.yaml').read_text()
            for role in ('reward','step','val'):
                self.assertRegex(text, rf'(?m)^{role}_api_provider: "vendor_a"$')
                self.assertRegex(text, rf'(?m)^{role}_api_key: null')
                self.assertRegex(text, rf'(?m)^{role}_api_fallback_models: null$')
    def test_judge_model_env_reaches_manager(self):
        text=(ROOT/'verl/workers/reward_manager/api_prime_dapo_trajectory_v3_gig.py').read_text()
        self.assertIn('KNOWGPO_JUDGE_MODEL',text)
        self.assertIn('KNOWGPO_JUDGE_API_KEY',text)
    def test_launcher_dry_run_does_not_reveal_auth(self):
        result=subprocess.run(['bash','scripts/train_knowgpo.sh','--dry-run'],cwd=ROOT,
               env={'PATH':'/usr/bin:/bin','KNOWGPO_JUDGE_API_KEY':'SENSITIVE_TEST_SECRET'},
               capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('SENSITIVE_TEST_SECRET',result.stdout)

    def test_method_overlay_disables_length_penalty(self):
        text=(ROOT/'verl/trainer/config/reward_dapo_trajectory_v7_rgc_ir.yaml').read_text()
        self.assertRegex(text,r'(?m)^length_penalty_mode: "none"$')
        self.assertRegex(text,r'(?m)^advantage_fusion_mode: "convex_final_exempt"$')
        self.assertRegex(text,r'(?m)^process_reward_omega: 0.3$')
if __name__=='__main__':unittest.main()
