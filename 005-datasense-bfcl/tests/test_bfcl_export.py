import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('exporter',ROOT/'scripts/bfcl_board_export.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ExportTest(unittest.TestCase):
 def test_partial_result_not_scored_as_success(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p=root/'bfcl/random100-e1/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_base_result.json';p.parent.mkdir(parents=True)
   item={'id':'multi_turn_base_0','result':[['f()']],'inference_log':[{'begin_of_turn_query':[{'role':'user','content':'task'}],'step_0':[{'role':'assistant','content':'f()','reasoning_content':'reason'},{'role':'tool','content':'ok'}]}]}
   p.write_text(json.dumps(item)+'\n'+ '{"id":');m.Exporter(root).update()
   out=json.loads((root/'jobs/e1/multi_turn_base_0/result.json').read_text());self.assertIsNone(out['verifier_result']['rewards']['reward'])
   traj=json.loads((root/'jobs/e1/multi_turn_base_0/agent/trajectory.json').read_text());self.assertEqual([x['role'] for x in traj['steps']],['user','assistant','tool']);self.assertEqual(traj['steps'][1]['reasoning_content'],'reason')
   self.assertEqual(json.loads((root/'progress.json').read_text())['generated'],1)
 def test_scores_appear_only_after_verification(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src=root/'bfcl/random100-e1'
   for cat in m.CATS:
    raw=src/f'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{cat}_result.json';raw.parent.mkdir(parents=True,exist_ok=True)
    raw.write_text(''.join(json.dumps({'id':f'multi_turn_{cat}_{i}','result':[]})+'\n' for i in range(200)))
    score=src/f'score/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{cat}_score.json';score.parent.mkdir(parents=True,exist_ok=True)
    score.write_text(json.dumps({'total_count':200,'correct_count':199})+'\n'+json.dumps({'id':f'multi_turn_{cat}_0','valid':False})+'\n')
   exporter=m.Exporter(root);exporter.update();self.assertIsNone(json.loads((root/'jobs/e1/multi_turn_base_1/result.json').read_text())['verifier_result']['rewards']['reward'])
   (src/'verified-summary.json').write_text('{}');exporter.update()
   self.assertEqual(json.loads((root/'jobs/e1/multi_turn_base_0/result.json').read_text())['verifier_result']['rewards']['reward'],0)
   self.assertEqual(json.loads((root/'jobs/e1/multi_turn_base_1/result.json').read_text())['verifier_result']['rewards']['reward'],1)
   result=json.loads((root/'jobs/e1/result.json').read_text());self.assertEqual(len(result['stats']['evals']['bfcl']['reward_stats']['reward']['1']),796)
if __name__=='__main__':unittest.main()
