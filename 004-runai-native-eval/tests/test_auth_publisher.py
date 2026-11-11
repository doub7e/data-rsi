import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import sync_kubeconfig as publisher
import native_eval_controller as controller


class PublisherTests(unittest.TestCase):
    def config(self):
        return {'users':[{'name':'example','user':{'auth-provider':{'name':'oidc','config':{'id-token':'fake-id-token','refresh-token':'never-log-this-refresh'}},'exec':{'command':'refresh-command'},'password':'never-log-this-password'}}]}

    def test_consumers_have_no_refresh_or_exec_credentials(self):
        original=self.config();result=publisher.consumer_config(original)
        self.assertEqual(result['users'][0]['user'],{'token':'fake-id-token'})
        self.assertIn('auth-provider',original['users'][0]['user'])
        self.assertNotIn('never-log',json.dumps(result))
        with self.assertRaises(RuntimeError):publisher.consumer_config({'users':[]})

    def test_publisher_uses_explicit_source_and_logs_no_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'target';events=Path(tmp)/'events'
            with patch.object(publisher,'SOURCE',Path('/expected/source')),patch.object(publisher,'TARGET',target),patch.object(publisher,'EVENTS',events),patch.dict(os.environ,{'KUBECONFIG':'/wrong/consumer'}),patch.object(publisher.subprocess,'run') as call,patch.object(publisher.subprocess,'check_output',return_value=json.dumps(self.config()).encode()) as view:
                self.assertTrue(publisher.sync()['token_changed'])
                before=target.stat().st_mtime_ns
                self.assertFalse(publisher.sync()['token_changed'])
                self.assertEqual(target.stat().st_mtime_ns,before)
                self.assertEqual(call.call_args.kwargs['env']['KUBECONFIG'],'/expected/source')
                self.assertEqual(view.call_args.kwargs['env']['KUBECONFIG'],'/expected/source')
                self.assertEqual(target.stat().st_mode&0o777,0o600)
                self.assertNotIn('fake-id-token',events.read_text())
                self.assertNotIn('never-log',events.read_text())

    def test_second_publisher_cannot_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'token.json'
            with target.with_suffix('.refresh.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.object(publisher,'TARGET',target),patch.object(publisher,'sync') as sync:
                    self.assertEqual(publisher.main(once=True),75);sync.assert_not_called()

    def test_failure_logging_does_not_include_exception_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(publisher,'TARGET',root/'token.json'),patch.object(publisher,'STATUS',root/'status.json'),patch.object(publisher,'EVENTS',root/'events.jsonl'),patch.object(publisher,'sync',side_effect=RuntimeError('token=never-log-me')):
                self.assertEqual(publisher.main(once=True),1)
                self.assertFalse(json.loads((root/'status.json').read_text())['ok'])
                self.assertNotIn('never-log-me',(root/'events.jsonl').read_text())
                self.assertNotIn('never-log-me',(root/'status.json').read_text())

    def test_monitor_reattaches_without_touching_jobs_or_run_state(self):
        with patch.object(controller.c,'job_exists',return_value={'running':True}),patch.object(controller.c,'write') as write,patch.object(controller,'ensure_serving') as serve,patch.object(controller,'launch_worker') as launch:
            controller.start_or_attach({'worker_job':'worker'},Path('/config'),Path('/node'),Path('/run'))
            write.assert_not_called();serve.assert_not_called();launch.assert_not_called()


if __name__=='__main__':unittest.main()
