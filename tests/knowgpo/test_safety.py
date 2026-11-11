import ast
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]

class Safety(unittest.TestCase):
    def test_v3_config_is_package_relative(self):
        src = (ROOT / 'verl/workers/reward_manager/api_prime_dapo_trajectory_v3_gig.py').read_text()
        self.assertNotIn('"/root/verl/', src)
        self.assertIn('KNOWGPO_JUDGE_API_KEY', src)

    def test_credentials_are_not_baked_into_active_judge_path(self):
        for rel in ('verl/utils/reward_score/api_client_factory.py',
                    'verl/utils/reward_score/process_reward_gig.py',
                    'verl/trainer/config/reward_dapo_trajectory_v3_gig.yaml',
                    'verl/trainer/config/reward_dapo_trajectory_v4_rgc.yaml'):
            text = (ROOT / rel).read_text()
            self.assertNotRegex(text, r'[A-Za-z0-9]{15,}@[0-9]{4}', rel)
        self.assertIn('KNOWGPO_JUDGE_API_KEY', (ROOT / 'verl/utils/reward_score/api_client_factory.py').read_text())

if __name__ == '__main__':unittest.main()
