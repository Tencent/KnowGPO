"""Check V7's actual inherited Phase-B call and exception-safe restoration offline."""
import ast
import pathlib
import types
import unittest
from unittest.mock import patch

ROOT=pathlib.Path(__file__).resolve().parents[2]
SOURCE=ROOT/'verl/workers/reward_manager/api_prime_dapo_trajectory_v7_rgc_ir.py'
TREE=ast.parse(SOURCE.read_text())
METHOD=next(n for c in TREE.body if isinstance(c,ast.ClassDef) and c.name=='ApiPrimeDapoTrajectoryV7RGCIRRewardManager' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
class _V3Module(types.ModuleType):pass
class _V3:
 def __call__(self,data,return_dict=False):
  from verl.workers.reward_manager import api_prime_dapo_trajectory_v3_gig as v3
  assert v3.compute_process_reward_phase_b is not ORIGINAL
  params=dict(phase_a_result={'all_steps_per_sample':[[{'is_final':True}]],'corrected_state_vectors_per_sample':[[[0]]],'batch_size':1},trajectories=[''],questions=['q'],rubrics_list=[['fact']],outcome_rewards=[0],uid_list=['q'])
  result=v3.compute_process_reward_phase_b(**params)
  data.non_tensor_batch['gig_step_advantages']=result['step_advantages_per_sample']
  return {'reward_tensor':0}

def ORIGINAL(**kwargs):return None
V3MODULE=_V3Module('verl.workers.reward_manager.api_prime_dapo_trajectory_v3_gig')
V3MODULE.compute_process_reward_phase_b=ORIGINAL

def _phase_b(**kwargs):return {'step_advantages_per_sample':[[{'step_advantage':0}]]}

class Dispatch(unittest.TestCase):
 def test_uses_v7_phase_b_and_restores_parent_binding(self):
  import sys
  saved={k:sys.modules.get(k) for k in ('verl','verl.workers','verl.workers.reward_manager','verl.workers.reward_manager.api_prime_dapo_trajectory_v3_gig','numpy')}
  verl=types.ModuleType('verl');verl.__path__=[]
  workers=types.ModuleType('verl.workers');workers.__path__=[]
  verl.workers=workers
  sys.modules['verl']=verl
  sys.modules['verl.workers']=workers
  manager_pkg=types.ModuleType('verl.workers.reward_manager');manager_pkg.api_prime_dapo_trajectory_v3_gig=V3MODULE
  workers.reward_manager=manager_pkg
  sys.modules['verl.workers.reward_manager']=manager_pkg
  sys.modules['verl.workers.reward_manager.api_prime_dapo_trajectory_v3_gig']=V3MODULE
  numpy=types.ModuleType('numpy');numpy.array=lambda values,dtype=None:list(values);sys.modules['numpy']=numpy
  try:
   env={'Any':object,'_gig_mod':types.SimpleNamespace(compute_process_reward_phase_b=ORIGINAL),'compute_rgc_ir_phase_b':_phase_b,'ApiPrimeDapoTrajectoryV3GiGRewardManager':_V3,'logger':__import__('logging').getLogger('test')}
   exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.FunctionDef(name='call_v7',args=METHOD.args,body=METHOD.body,decorator_list=[],returns=None,type_comment=None)],type_ignores=[])),str(SOURCE),'exec'),env)
   data=types.SimpleNamespace(non_tensor_batch={'uid':['q']})
   v7=types.SimpleNamespace(config={'advantage_fusion_mode':'convex_final_exempt'})
   result=env['call_v7'](v7,data,return_dict=True)
   self.assertEqual(result['reward_tensor'],0)
   self.assertEqual(data.non_tensor_batch['gig_step_advantages'][0][0]['step_advantage'],0)
   self.assertEqual(data.non_tensor_batch['gig_advantage_fusion_mode'][0],'convex_final_exempt')
   self.assertIs(V3MODULE.compute_process_reward_phase_b,ORIGINAL)
  finally:
   for k,v in saved.items():
    if v is None:sys.modules.pop(k,None)
    else:sys.modules[k]=v
 def test_restores_binding_when_phase_b_raises(self):
  import sys
  saved={k:sys.modules.get(k) for k in ('verl','verl.workers','verl.workers.reward_manager','verl.workers.reward_manager.api_prime_dapo_trajectory_v3_gig','numpy')}
  verl=types.ModuleType('verl');verl.__path__=[]
  workers=types.ModuleType('verl.workers');workers.__path__=[]
  manager_pkg=types.ModuleType('verl.workers.reward_manager');manager_pkg.__path__=[]
  verl.workers=workers;workers.reward_manager=manager_pkg;manager_pkg.api_prime_dapo_trajectory_v3_gig=V3MODULE
  for k,v in (('verl',verl),('verl.workers',workers),('verl.workers.reward_manager',manager_pkg),('verl.workers.reward_manager.api_prime_dapo_trajectory_v3_gig',V3MODULE)):
   sys.modules[k]=v
  def fail(**kwargs):raise ValueError('judge offline')
  env={'Any':object,'_gig_mod':types.SimpleNamespace(compute_process_reward_phase_b=ORIGINAL),'compute_rgc_ir_phase_b':fail,'ApiPrimeDapoTrajectoryV3GiGRewardManager':_V3,'logger':__import__('logging').getLogger('test')}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.FunctionDef(name='call_v7',args=METHOD.args,body=METHOD.body,decorator_list=[],returns=None,type_comment=None)],type_ignores=[])),str(SOURCE),'exec'),env)
  try:
   with self.assertRaisesRegex(ValueError,'judge offline'):
    env['call_v7'](types.SimpleNamespace(config={'advantage_fusion_mode':'convex_final_exempt'}),types.SimpleNamespace(non_tensor_batch={'uid':['q']}))
   self.assertIs(V3MODULE.compute_process_reward_phase_b,ORIGINAL)
   self.assertIs(env['_gig_mod'].compute_process_reward_phase_b,ORIGINAL)
  finally:
   for k,v in saved.items():
    if v is None:sys.modules.pop(k,None)
    else:sys.modules[k]=v

if __name__=='__main__':unittest.main()
