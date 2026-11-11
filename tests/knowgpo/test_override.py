import pathlib
import subprocess
import unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
class Overrides(unittest.TestCase):
 def test_paper_batch_override_appears_once(self):
  proc=subprocess.run(['bash','scripts/train_knowgpo.sh','--dry-run','data.train_batch_size=32'],cwd=ROOT,capture_output=True,text=True)
  self.assertEqual(proc.returncode,0,proc.stderr)
  self.assertEqual(proc.stdout.count('data.train_batch_size='),1)
  self.assertIn('data.train_batch_size=32',proc.stdout)
 def test_launcher_rejects_other_reward_managers(self):
  proc=subprocess.run(['bash','scripts/train_knowgpo.sh','--dry-run','reward_model.reward_manager=naive'],cwd=ROOT,capture_output=True,text=True)
  self.assertNotEqual(proc.returncode,0)
  self.assertIn('KnowGPO',proc.stderr)
if __name__=='__main__':unittest.main()
