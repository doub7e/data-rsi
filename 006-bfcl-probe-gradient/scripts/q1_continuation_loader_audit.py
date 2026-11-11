"""Read installed immutable-image sources to verify continuation weight loading."""
import inspect,importlib.metadata
from common import *
run=ROOT/'runs/q1c-loader-audit-20261104';state(run,'inspect_training_loader')
import llamafactory.model.loader as loader
import llamafactory.model.patcher as patcher
import llamafactory.model.adapter as adapter
import llamafactory.hparams.parser as parser
import llamafactory.hparams.model_args as model_args
import llamafactory.hparams.finetuning_args as finetuning_args
for name,module in [('loader',loader),('patcher',patcher),('adapter',adapter),('hparams_parser',parser),('model_args',model_args),('finetuning_args',finetuning_args)]:
 (run/(name+'.py')).write_text(inspect.getsource(module))
write(run/'result.json',{'version':importlib.metadata.version('llamafactory'),'sources':{n:sha(run/(n+'.py')) for n in ['loader','patcher','adapter','hparams_parser']}});from q1_continuation_data import random50
random50()
write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded')
