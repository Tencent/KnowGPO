"""Verify the actual reward metadata hand-off used by remote reward execution."""
import ast
import pathlib
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[2]
tree=ast.parse((ROOT/'verl/trainer/ppo/reward.py').read_text())
fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='compute_reward')

class RewardMetadata(unittest.TestCase):
 def test_remote_reward_returns_v7_fusion_hint(self):
  class Batch:
   non_tensor_batch={'gig_step_advantages':[[{'step_advantage':1.0}]],
                     'gig_omega':[0.3],
                     'gig_advantage_fusion_mode':['convex_final_exempt']}
  class RewardFn:
   def __call__(self,data,return_dict=False):
    return {'reward_tensor':0,'reward_extra_info':{}}
  namespace={'AbstractRewardManager':object,'DataProto':object,'Any':object,'torch':type('Torch',(),{'Tensor':object})}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/'verl/trainer/ppo/reward.py'),'exec'),namespace)
  _,extra=namespace['compute_reward'](Batch(),RewardFn())
  self.assertEqual(extra['gig_advantage_fusion_mode'],['convex_final_exempt'])

if __name__=='__main__':unittest.main()
