"""Offline invariants for the single KnowGPO V7 training path."""
import ast
import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class TrainingChain(unittest.TestCase):
    def test_reward_inheritance_and_registration_resolve(self):
        reward = ROOT / 'verl/workers/reward_manager'
        for version in ('v3_gig', 'v4_rgc', 'v7_rgc_ir'):
            path = reward / f'api_prime_dapo_trajectory_{version}.py'
            self.assertTrue(path.is_file(), path)
            ast.parse(path.read_text())
        registry = (reward / '__init__.py').read_text()
        for version in ('v3_gig', 'v4_rgc', 'v7_rgc_ir'):
            self.assertIn(f'from .api_prime_dapo_trajectory_{version} import', registry)
        for version in ('v1_naive', 'v2_cgrpo', 'v5_gigpo', 'v6_rgc_gigpo', 'v8_igpo'):
            self.assertNotIn(f'from .api_prime_dapo_trajectory_{version} import', registry)

    def test_hydra_ppo_composition_exists(self):
        base = ROOT / 'verl/trainer/config'
        for part in ('__init__.py', 'config.py', 'algorithm.py', 'ppo_trainer.yaml',
                     'actor/dp_actor.yaml', 'ref/dp_ref.yaml', 'rollout/rollout.yaml',
                     'data/legacy_data.yaml', 'critic/dp_critic.yaml',
                     'reward_model/dp_reward_model.yaml', 'reward_model/reward_model.yaml',
                     'reward_dapo_trajectory_v3_gig.yaml',
                     'reward_dapo_trajectory_v4_rgc.yaml',
                     'reward_dapo_trajectory_v7_rgc_ir.yaml'):
            self.assertTrue((base / part).is_file(), part)

    def test_tool_registry_is_python_and_class_exists(self):
        import re
        ast.parse((ROOT / 'verl/tools/utils/tool_registry.py').read_text())
        config = (ROOT / 'verl/tools/search_tool_config.yaml').read_text()
        m = re.search(r'^\s*- class_name:\s*(verl\.tools\.[A-Za-z_]+)\.SearchTool$', config, re.M)
        self.assertIsNotNone(m)
        self.assertTrue((ROOT / pathlib.Path(*m.group(1).split('.')).with_suffix('.py')).exists())

    def test_v7_launcher_dry_run_has_no_side_effects(self):
        proc = subprocess.run(['bash', 'scripts/train_knowgpo.sh', '--dry-run'], cwd=ROOT,
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('apiprimedapotrajectoryv7rgcir', proc.stdout)
        self.assertIn('verl.trainer.main_ppo', proc.stdout)
        self.assertNotIn('ray job submit', proc.stdout)


if __name__ == '__main__':
    unittest.main()
