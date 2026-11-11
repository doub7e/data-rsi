import importlib.util,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eval'))
import pytest
spec=importlib.util.spec_from_file_location('eval_pipeline',Path(__file__).resolve().parents[1]/'eval/pipeline.py')
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)

def result_file(tmp_path,**changes):
 d={'finished_at':'2026-10-30T00:00:00Z','n_total_trials':100,'stats':{'n_completed_trials':90,'n_errored_trials':10,'n_running_trials':0,'n_pending_trials':0,'n_cancelled_trials':0,'evals':{'model':{'reward_stats':{'reward':{'1.0':[str(i) for i in range(25)],'0.0':['fail']}},'exception_stats':{'AgentTimeoutError':['error']*10}}}}}
 d.update(changes);p=tmp_path/'result.json';p.write_text(json.dumps(d));return p

def test_fixed_denominator_counts_errors_as_zero(tmp_path):
 d=pipeline.trial_summary(result_file(tmp_path),100)
 assert d['accuracy_pct']==25.0
 assert d['errors']['AgentTimeoutError']==10

def test_missing_tasks_are_not_a_valid_comparison(tmp_path):
 with pytest.raises(RuntimeError,match='denominator'):pipeline.trial_summary(result_file(tmp_path,n_total_trials=99),100)

def test_incomplete_run_is_not_reported_as_final(tmp_path):
 with pytest.raises(RuntimeError,match='not complete'):pipeline.trial_summary(result_file(tmp_path,finished_at=None),100)

def test_repeat_standard_error():
 d=pipeline.aggregate([{'accuracy_pct':10},{'accuracy_pct':20},{'accuracy_pct':30}])
 assert d['accuracy_pct']==20
 assert d['se_pct']==pytest.approx(10/math.sqrt(3))
 with pytest.raises(RuntimeError):pipeline.aggregate([{'accuracy_pct':10}])
