"""Run only the real stdlib-dependent V7 math function without GPU dependencies."""
import ast
import math
import pathlib
import unittest
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]

class _Array:
    def __init__(self,values):self.values=list(values)
    def __len__(self):return len(self.values)
    def mean(self):return sum(self.values)/len(self.values)
    def std(self,ddof=0):
        avg=self.mean()
        return math.sqrt(sum((x-avg)**2 for x in self.values)/(len(self.values)-ddof))
class _Np:
    float64=float
    max=staticmethod(max)
    min=staticmethod(min)
    @staticmethod
    def asarray(values,dtype=None):return _Array(values)
    @staticmethod
    def mean(values):return _Array(values).mean()
    @staticmethod
    def std(values,ddof=0):return _Array(values).std(ddof)

source=(ROOT/'verl/utils/reward_score/process_reward_rgc.py').read_text()
tree=ast.parse(source)
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_safe_mean_std','_safe_mean_std_unbiased','compute_rgc_ir_step_advantages')]
namespace={'List':list,'Dict':dict,'Any':object,'Optional':__import__('typing').Optional,'Tuple':tuple,'defaultdict':defaultdict,'np':_Np(),'logger':__import__('logging').getLogger('offline')}
exec(compile(ast.Module(body=selected,type_ignores=[]),str(ROOT/'verl/utils/reward_score/process_reward_rgc.py'),'exec'),namespace)
compute=namespace['compute_rgc_ir_step_advantages']

class V7Math(unittest.TestCase):
    def test_pre_action_grouping_increment_and_terminal_exemption(self):
        steps=[[{'step_index':0,'is_final':False},{'step_index':1,'is_final':True}],
               [{'step_index':0,'is_final':False},{'step_index':1,'is_final':True}]]
        records, metrics, groups=compute(steps,[[[1],[1]],[[0],[0]]],[1,0],['q','q'])
        self.assertAlmostEqual(records[0][0]['step_advantage'], 2**-0.5,places=6)
        self.assertAlmostEqual(records[1][0]['step_advantage'],-2**-0.5,places=6)
        self.assertEqual([row[1]['step_advantage'] for row in records],[0,0])
        self.assertEqual([row[0]['corrected_state'] for row in records],[[0],[0]])
    def test_different_questions_do_not_share_knowledge_group(self):
        steps=[[{'is_final':False}], [{'is_final':False}]]
        records,_,_=compute(steps,[[[1]],[[0]]],[1,0],['q1','q2'])
        self.assertEqual([row[0]['step_advantage'] for row in records],[0,0])
    def test_constant_increment_has_no_group_signal(self):
        steps=[[{'is_final':False}], [{'is_final':False}]]
        records,_,_=compute(steps,[[[1]],[[1]]],[1,1],['q','q'])
        self.assertEqual([row[0]['step_advantage'] for row in records],[0,0])

    def test_changes_after_first_step_change_group_key(self):
        steps=[[{'is_final':False},{'is_final':False},{'is_final':True}],
               [{'is_final':False},{'is_final':False},{'is_final':True}]]
        states=[[[1,0],[1,1],[1,1]],[[0,0],[0,0],[0,0]]]
        records,_,_=compute(steps,states,[1,0],['q','q'])
        self.assertEqual(records[0][1]['corrected_state'],[1,0])
        self.assertEqual(records[1][1]['corrected_state'],[0,0])
        self.assertNotEqual(records[0][1]['group_id'],records[1][1]['group_id'])
        self.assertEqual([row[-1]['step_advantage'] for row in records],[0,0])

if __name__=='__main__':unittest.main()
