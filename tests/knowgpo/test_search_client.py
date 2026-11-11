"""Search client must accept generic endpoint contract without a private adapter."""
import ast
import pathlib
import unittest
from unittest.mock import patch

ROOT=pathlib.Path(__file__).resolve().parents[2]
FILE=ROOT/'verl/tools/utils/search_r1_like_utils.py'
TREE=ast.parse(FILE.read_text())
NODE=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='call_search_api')
class Response:
 status_code=200
 def raise_for_status(self):pass
 def json(self):return {'result':[[{'title':'Example','url':'https://example.org','snippet':'found'}]]}
class Requests:
 class exceptions:
  ConnectionError=OSError
  Timeout=TimeoutError
  RequestException=Exception
 def post(self,url,headers,json,timeout):
  if url!='https://search.example.org/api':raise AssertionError('wrong endpoint')
  if json!={'queries':['history'],'topk':10,'return_scores':True}:raise AssertionError('wrong payload')
  return Response()
class TestSearch(unittest.TestCase):
 def test_generic_post_payload(self):
  import uuid,logging
  n={'Optional':__import__('typing').Optional,'Any':object,'uuid':uuid,'logger':logging.getLogger('test'),'requests':Requests(),
     'time':__import__('time'),'DEFAULT_TIMEOUT':30,'MAX_RETRIES':1,'INITIAL_RETRY_DELAY':1}
  exec(compile(ast.Module(body=[NODE],type_ignores=[]),str(FILE),'exec'),n)
  body,error=n['call_search_api']('https://search.example.org/api',['history'],topk=10)
  self.assertEqual(body['result'][0][0]['title'],'Example')
  self.assertIsNone(error)
if __name__=='__main__':unittest.main()
