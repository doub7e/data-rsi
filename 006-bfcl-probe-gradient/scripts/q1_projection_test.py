from common import *
from q1_projection_math import unit_checks
if __name__=='__main__':
 run=ROOT/'runs/q1p-numerics-20261105';state(run,'numerical_checks')
 try:
  result=unit_checks();write(run/'result.json',result);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded');print(json.dumps(result),flush=True)
 except BaseException as e:state(run,'failed','failed',error=str(e));raise
