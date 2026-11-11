"""Offline arithmetic of the production V7 fusion branch, no import-time GPU effects."""
import ast
import pathlib
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[2]
TREE=ast.parse((ROOT/'verl/trainer/ppo/ray_trainer.py').read_text())

def find_v7_branch(node):
 for child in ast.walk(node):
  if isinstance(child,ast.If) and isinstance(child.test,ast.Compare):
   try: expression=ast.unparse(child.test)
   except Exception:continue
   if expression=="_fusion_mode == 'convex_final_exempt'":return child.body
 raise AssertionError('V7 fusion branch not found')

BRANCH=find_v7_branch(TREE)

class Vec:
 def __init__(self,values):self.v=values
 def __mul__(self,x):return Vec([i*x for i in self.v])
 __rmul__=__mul__
 def __add__(self,x):return Vec([a+b for a,b in zip(self.v,x.v)])
 def __eq__(self,other):return self.v==(other.v if isinstance(other,Vec) else other)
class FakeNp:
 @staticmethod
 def where(mask,a,b):return Vec([x if m else y for m,x,y in zip(mask,a.v,b.v)])

class Fusion(unittest.TestCase):
 def test_nonfinal_convex_and_final_raw_episode(self):
  source=ast.fix_missing_locations(ast.Module(body=BRANCH,type_ignores=[]))
  ctx={'np':FakeNp,'_fusion_mode':'convex_final_exempt',
       'omega':0.3,'episode_advantages_norm':Vec([2.0,2.0]),
       'step_adv_norm':Vec([1.0,0.0]),'episode_advantages':Vec([5.0,5.0]),
       'final_token_mask':[False,True]}
  exec(compile(source,str(ROOT/'verl/trainer/ppo/ray_trainer.py'),'exec'),ctx)
  self.assertAlmostEqual(ctx['fused_advantages'].v[0],1.7)
  self.assertEqual(ctx['fused_advantages'].v[1],5.0)

if __name__=='__main__':unittest.main()
