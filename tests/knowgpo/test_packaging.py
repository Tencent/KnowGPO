import pathlib
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[2]
class InstallMetadata(unittest.TestCase):
 def test_package_metadata_and_install_instructions_exist(self):
  self.assertTrue((ROOT/'pyproject.toml').is_file())
  self.assertIn('verl',(ROOT/'pyproject.toml').read_text())
  self.assertTrue((ROOT/'README.md').is_file())
 def test_launch_configuration_requires_external_inputs(self):
  p=ROOT/'scripts/train_knowgpo.sh';self.assertTrue(p.is_file());text=p.read_text()
  for variable in ('KNOWGPO_MODEL_PATH','KNOWGPO_TRAIN_FILES','KNOWGPO_VAL_FILES','KNOWGPO_SEARCH_URL','KNOWGPO_JUDGE_API_KEY','KNOWGPO_JUDGE_BASE_URL'):
   self.assertIn(variable,text)
if __name__=='__main__':unittest.main()
