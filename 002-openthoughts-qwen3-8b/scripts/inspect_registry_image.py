"""Read the published training image digest; never prints registry credentials."""
import json
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

HOST='registry.rcp.epfl.ch'
REPO='shuli/data-rsi-sft'
TAG='qwen3-cu128-torch280-v1'
ROOT=Path(__file__).resolve().parents[1]
config=Path('/mnt/cvlab/scratch/cvlab/home/shuli/.docker/config.json')
auth=json.loads(config.read_text())['auths'][HOST]['auth']
authorization='Basic '+auth
accept='application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json'
def request(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'Authorization':authorization,'Accept':accept}),timeout=30)
url=f'https://{HOST}/v2/{REPO}/manifests/{TAG}'
try:
    response=request(url)
except urllib.error.HTTPError as e:
    if e.code!=401: raise
    fields=dict(re.findall(r'(\w+)="([^"]+)"',e.headers['WWW-Authenticate']))
    realm=fields['realm']
    parsed=urllib.parse.urlsplit(realm)
    if parsed.hostname!=HOST or parsed.scheme!='https': raise RuntimeError('Unexpected token realm')
    with request(realm+'?'+urllib.parse.urlencode({'service':fields.get('service',HOST),'scope':f'repository:{REPO}:pull'})) as token_response:
        token=json.load(token_response)
    authorization='Bearer '+(token.get('token') or token['access_token'])
    response=request(url)
with response:
    digest=response.headers['Docker-Content-Digest']
    manifest=json.load(response)
result={'image':f'{HOST}/{REPO}:{TAG}','digest':digest,'immutable_image':f'{HOST}/{REPO}@{digest}','compressed_layer_bytes':sum(x['size'] for x in manifest.get('layers',[])),'manifest':manifest}
(ROOT/'evidence/image-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='manifest'},indent=2))
