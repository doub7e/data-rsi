import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import shlex
import subprocess
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'native'))
import native_common as common
import native_eval_worker as worker
import prepare_all
import resume_native_job
import native_eval_controller as controller
import auth_guard
from datetime import datetime,timezone,timedelta


class ProductionTests(unittest.TestCase):
    def test_complete_swe_can_start_while_other_images_build(self):
        with patch.object(common,'benchmark_ready',side_effect=lambda name:name=='swebench100'):
            self.assertTrue(common.initial_images_ready())
        with patch.object(common,'benchmark_ready',return_value=False):
            self.assertFalse(common.initial_images_ready())

    def test_original_copy_context_has_no_injected_control_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';(source/'environment').mkdir(parents=True)
            (source/'environment/Dockerfile').write_text('FROM ubuntu:24.04\nCOPY . /app\n')
            (source/'environment/input.txt').write_text('fixture')
            with patch.object(prepare_all,'ROOT',root):
                context=prepare_all.context_for(source,'test',{})
            self.assertEqual(sorted(p.name for p in context.iterdir()),['Dockerfile','input.txt'])
            self.assertIn("RUN printf '%s\\n'",(context/'Dockerfile').read_text())
            command=(context/'Dockerfile').read_text().strip().splitlines()[-1].removeprefix('RUN ')
            target=root/'rendered-entrypoint.sh'
            command=command.replace('/usr/local/bin/harbor-sandbox-entrypoint',shlex.quote(str(target)))
            subprocess.run(['/bin/sh','-c',command],check=True)
            subprocess.run(['/bin/bash','-n',str(target)],check=True)
            self.assertIn("trap 'exit 0' TERM INT\n",target.read_text())

    def test_pending_build_authorization_cannot_submit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(prepare_all,'ROOT',Path(tmp)), \
             patch.object(prepare_all,'call') as call, patch.object(prepare_all,'registry') as registry:
            with self.assertRaisesRegex(RuntimeError,'approval is pending'):prepare_all.main()
            call.assert_not_called();registry.assert_not_called()

    def test_nonfinal_checkpoint_is_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            common.write(root/'result.json',{'status':'succeeded','epoch':6.0})
            common.write(root/'state.json',{'state':'succeeded','exit_code':0})
            self.assertFalse(common.training_complete(root))

    def test_all_native_configs_preserve_json_and_model_identity(self):
        for scale in ['1k','10k']:
            config=common.read(ROOT/f'production/eval-{scale}.json')
            with patch.dict('os.environ',{'NATIVE_MODEL_API_BASE':'http://model/v1'}):
                cfg=worker.job_config(config,Path('/tmp/run'),'example',Path('/tmp/tasks'))
            self.assertEqual(cfg['environment']['import_path'],'runai_env:RunAIEnvironment')
            self.assertEqual(cfg['n_concurrent_trials'],32)
            self.assertEqual(cfg['agents'][0]['kwargs']['parser_name'],'json')
            self.assertEqual(cfg['agents'][0]['model_name'],'hosted_vllm/'+config['model_name'])
            self.assertNotIn('daytona',json.dumps(cfg).lower())

    def test_failed_trials_remain_in_fixed_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json'
            common.write(path,{'finished_at':'now','n_total_trials':100,
                'stats':{'evals':{'model':{'reward_stats':{'reward':{'1.0':['one']}},
                                           'exception_stats':{'AgentTimeoutError':['bad']}}}}})
            self.assertEqual(common.trial_summary(path,100)['accuracy_pct'],1)
            with self.assertRaisesRegex(RuntimeError,'denominator'):common.trial_summary(path,99)

    def test_resume_preserves_errors_and_archives_stale_root_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp)/'jobs/batch'
            for name in ['passed','failed','retry']:
                trial=job/name
                common.write(trial/'config.json',{})
                common.write(trial/'result.json',{'finished_at':'now','exception_info':{'exception_type':'RuntimeError'} if name!='passed' else None})
                common.write(trial/'attempts/000/result.json',{'finished_at':'now'})
            (job/'retry/attempts/001').mkdir()
            prior={name:(job/name/'result.json').read_bytes() for name in ['passed','failed']}
            preserved,archive=resume_native_job.prepare_resume(job)
            self.assertEqual(set(preserved),{'passed','failed'})
            self.assertFalse((job/'retry').exists())
            self.assertTrue((archive/'retry/attempts/001').exists())
            for name,value in prior.items():self.assertEqual((job/name/'result.json').read_bytes(),value)

    def test_authorized_parallel_evaluation_has_a_run_scoped_lock(self):
        run=Path('/experiment/runs/eval-10k')
        self.assertEqual(controller.evaluation_lock_path({},run),common.ROOT.parent/'.native-evaluation.lock')
        self.assertEqual(controller.evaluation_lock_path({'parallel_evaluation':True},run),run/'.controller.lock')
        one=common.read(ROOT/'production/eval-1k.json');ten=common.read(ROOT/'production/eval-10k.json')
        for field in ['serve_job','worker_job','model_name','run_id']:
            self.assertNotEqual(one[field],ten[field])
        self.assertTrue(ten['parallel_evaluation'])

    def test_auth_guard_rejects_expired_missing_and_stale_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'auth.json';now=datetime.now(timezone.utc)
            self.assertFalse(auth_guard.available(p,now))
            for ok,age,expected in [(True,5,True),(False,5,False),(True,31,False)]:
                common.write(p,{'ok':ok,'updated_at':(now-timedelta(seconds=age)).isoformat()})
                self.assertEqual(auth_guard.available(p,now),expected)

    def test_auth_guard_prevents_process_launch(self):
        with patch.dict('os.environ',{'NATIVE_AUTH_GUARD':'1'}), patch.object(auth_guard,'available',return_value=False), patch.object(worker.subprocess,'Popen') as launch:
            with self.assertRaises(auth_guard.AuthenticationUnavailable):worker.run_harbor({},[],None)
            launch.assert_not_called()

    def test_selective_auth_recovery_preserves_model_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp)/'jobs/batch'
            cases={'auth':{'exception_type':'RuntimeError','exception_message':'Native sandbox cleanup failed: Unauthorized'},'timeout':{'exception_type':'AgentTimeoutError','exception_message':'Agent timed out'},'cancelled':{'exception_type':'CancelledError'}}
            for name,error in cases.items():
                common.write(job/name/'config.json',{})
                common.write(job/name/'result.json',{'finished_at':'now','exception_info':error})
            self.assertTrue(worker.needs_recovery(job))
            preserved,archive=resume_native_job.prepare_resume(job)
            self.assertEqual(set(preserved),{'timeout'})
            self.assertTrue((archive/'auth/result.json').exists())
            self.assertTrue((archive/'cancelled/result.json').exists())
            self.assertFalse(worker.needs_recovery(job))

    def test_legacy_auth_trace_is_recoverable_even_if_stderr_hid_reason(self):
        self.assertTrue(auth_guard.recoverable_result({'exception_info':{'exception_type':'RuntimeError','exception_message':'CLI v1 is now deprecated','exception_traceback':'return await self.run(argv, _auth_attempt=_auth_attempt+1)'}}))
        self.assertFalse(auth_guard.recoverable_result({'exception_info':{'exception_type':'RuntimeError','exception_message':'model parser failed'}}))

    def test_remote_task_auth_text_is_not_cluster_auth_failure(self):
        self.assertFalse(auth_guard.control_auth_error(['kubectl','exec'],'HTTP 401 Unauthorized','remote test failed: Unauthorized'))
        self.assertTrue(auth_guard.control_auth_error(['kubectl','get'],'','error: You must be logged in to the server (Unauthorized)'))
        self.assertTrue(auth_guard.control_auth_error(['runai','submit'],'Unauthorized','CLI deprecated'))

    def test_batch_auth_interruption_retries_without_advancing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);out=run/'jobs/batch'
            with patch.object(worker,'wait_for_auth'), patch.object(common,'state'), patch.object(worker,'cleanup') as cleanup, patch.object(worker,'needs_recovery',return_value=False), patch.object(worker,'run_harbor',side_effect=[auth_guard.AuthenticationUnavailable('expired'),0]) as harbor:
                worker.execute_batch({},run,'batch',run/'config.json',out)
            self.assertEqual(harbor.call_count,2);cleanup.assert_called_once()

    def test_resume_after_serving_replacement_updates_only_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);cfg={'agents':[{'model_name':'hosted_vllm/model','kwargs':{'api_base':'http://old/v1','temperature':.6}}],'timeout_multiplier':2}
            common.write(out/'config.json',cfg)
            worker.refresh_resume_endpoint(out,'http://new/v1','model')
            updated=common.read(out/'config.json');self.assertEqual(updated['agents'][0]['kwargs']['api_base'],'http://new/v1');self.assertEqual(updated['timeout_multiplier'],2);self.assertEqual(updated['agents'][0]['kwargs']['temperature'],.6)
            with self.assertRaises(RuntimeError):worker.refresh_resume_endpoint(out,'http://wrong/v1','different-model')

    def test_replacement_pod_prevents_workload_cleanup(self):
        failed={'status':{'phase':'Failed'}}
        self.assertTrue(controller.worker_terminal([failed]))
        self.assertFalse(controller.worker_terminal([]))
        for phase in ['Pending','Running']:
            self.assertFalse(controller.worker_terminal([failed,{'status':{'phase':phase}}]))

    def test_infrastructure_recovery_does_not_retry_arbitrary_failures(self):
        for message in ['Native evaluation interrupted','Model API unavailable; stopping native evaluation']:
            self.assertTrue(controller.recoverable_worker_failure({'state':'failed','message':message}))
        self.assertFalse(controller.recoverable_worker_failure({'state':'cancelled','message':'Native evaluation interrupted'}))
        self.assertFalse(controller.recoverable_worker_failure({'state':'failed','message':'invalid dataset'}))
        self.assertTrue(auth_guard.recoverable_result({'exception_info':{'exception_type':'APIConnectionError'}}))
        self.assertFalse(auth_guard.recoverable_result({'exception_info':{'exception_type':'AgentTimeoutError'}}))


if __name__=='__main__':unittest.main()
