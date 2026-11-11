import os
import pathlib
import subprocess
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'scripts/train_knowgpo.sh'

class Launcher(unittest.TestCase):
    def test_missing_parameters_rejected_before_training(self):
        result=subprocess.run(['bash',str(SCRIPT)],cwd=ROOT,env={'PATH':os.environ['PATH']},capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('KNOWGPO_MODEL_PATH',result.stderr)
    def test_dry_run_is_side_effect_free_and_v7_only(self):
        result=subprocess.run(['bash',str(SCRIPT),'--dry-run'],cwd=ROOT,env={'PATH':os.environ['PATH']},capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('reward_model.reward_manager=apiprimedapotrajectoryv7rgcir',result.stdout)
        self.assertIn('actor_rollout_ref.rollout.n=16',result.stdout)
        self.assertNotIn('ray job submit',result.stdout)

if __name__=='__main__':unittest.main()
