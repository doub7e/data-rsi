import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eval'))
from gates import require_build_ready,TrainingNotReady

@pytest.fixture
def environment(tmp_path):
    root=tmp_path/'node/eval';root.mkdir(parents=True)
    approval={'allowed':True,'start_after':'training_succeeded','max_concurrent':2,'max_new_images':200}
    (root/'build-authorization.json').write_text(json.dumps(approval))
    run=root.parent/'runs/repro-10k-4h200-20261030';(run/'checkpoint').mkdir(parents=True)
    (run/'result.json').write_text(json.dumps({'status':'succeeded','epoch':7.0}))
    (run/'state.json').write_text(json.dumps({'state':'succeeded','exit_code':0}))
    (run/'checkpoint/model.safetensors.index.json').write_text(json.dumps({'weight_map':{'weight':'model-1.safetensors'}}))
    (run/'checkpoint/model-1.safetensors').write_bytes(b'test fixture')
    return root,run

def test_authorized_complete_run_can_proceed(environment):
    root,_=environment
    assert require_build_ready(root)['max_concurrent']==2

def test_permission_alone_cannot_start_builds_during_training(environment):
    root,run=environment
    (run/'state.json').write_text('{"state":"running"}')
    with pytest.raises(TrainingNotReady):require_build_ready(root)

def test_incomplete_epochs_cannot_start_builds(environment):
    root,run=environment
    (run/'result.json').write_text('{"status":"succeeded","epoch":6}')
    with pytest.raises(TrainingNotReady):require_build_ready(root)

def test_missing_checkpoint_shard_blocks_builds(environment):
    root,run=environment
    (run/'checkpoint/model-1.safetensors').unlink()
    with pytest.raises(TrainingNotReady):require_build_ready(root)

def test_missing_approval_blocks_builds(environment):
    root,_=environment
    (root/'build-authorization.json').unlink()
    with pytest.raises(RuntimeError,match='authorization'):require_build_ready(root)

def test_concurrency_cannot_exceed_user_approval(environment):
    root,_=environment
    p=root/'build-authorization.json';d=json.loads(p.read_text());d['max_concurrent']=3;p.write_text(json.dumps(d))
    with pytest.raises(RuntimeError,match='concurrency'):require_build_ready(root)
