import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('resume',ROOT/'scripts/bfcl_resume.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ResumeTest(unittest.TestCase):
 def test_keep_model_outcomes_and_retry_infrastructure_only(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p=root/'bfcl/random100-e1/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_base_result.json';p.parent.mkdir(parents=True)
   data=[{'id':'ok','result':[]},{'id':'context','traceback':'openai.BadRequestError: maximum context length exceeded'},{'id':'infra','traceback':'openai.APIConnectionError: Connection error.'}]
   original=b''.join(json.dumps(r).encode()+b'\n' for r in data)+b'{"id":'
   p.write_bytes(original);marker=p.parents[2]/'verified-summary.json';marker.write_text('{}')
   launcher=m.prepare_runtime(root,'attempt2',ROOT/'upstream/code/bfcl/post-training')
   self.assertTrue(launcher.exists());self.assertFalse(marker.exists())
   self.assertEqual([json.loads(x)['id'] for x in p.read_text().splitlines()],['ok','context'])
   archive=root/'recovery/attempt2'/p.relative_to(root);self.assertEqual(archive.read_bytes(),original)
   adapted=(root/'harness-runtime/eval.py').read_text();self.assertNotIn("'--skip-server-setup','-o'",adapted)
   self.assertIn("'--skip-server-setup','-o'",(ROOT/'upstream/code/bfcl/post-training/eval.py').read_text())
   preserved=p.read_bytes();m.prepare_runtime(root,'attempt3',ROOT/'upstream/code/bfcl/post-training');self.assertEqual(p.read_bytes(),preserved)
 def test_reject_non_tail_corruption(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p=root/'bfcl/random100-e1/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_base_result.json';p.parent.mkdir(parents=True);p.write_bytes(b'broken\n{"id":"ok"}\n')
   with self.assertRaises(RuntimeError):m.prepare_runtime(root,'bad',ROOT/'upstream/code/bfcl/post-training')
if __name__=='__main__':unittest.main()
