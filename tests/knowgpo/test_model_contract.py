import ast
import pathlib
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[2]
class RuntimeContract(unittest.TestCase):
 def test_training_entrypoint_skips_optional_missing_datagen_module(self):
  source=(ROOT/'verl/trainer/main_ppo.py').read_text()
  self.assertNotIn('from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset',source)
  self.assertIn('verl.experimental.dynamic_dataset.dynamicgen_dataset',source)
 def test_v7_manager_has_no_fixed_remote_config_lookup(self):
  for name in ('api_prime_dapo_trajectory_v3_gig.py','api_prime_dapo_trajectory_v4_rgc.py','api_prime_dapo_trajectory_v7_rgc_ir.py'):
   s=(ROOT/'verl/workers/reward_manager'/name).read_text()
   self.assertNotIn('"/root/',s)
 def test_missing_training_inputs_do_not_have_deceptive_data_defaults(self):
  self.assertIn('train_files: null',(ROOT/'verl/trainer/config/data/legacy_data.yaml').read_text())
  self.assertIn('val_files: null',(ROOT/'verl/trainer/config/data/legacy_data.yaml').read_text())
  self.assertIn('path: null',(ROOT/'verl/trainer/config/ppo_trainer.yaml').read_text())
if __name__=='__main__':unittest.main()
