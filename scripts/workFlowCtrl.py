import httpx
import hmac, hashlib, argparse, json
from pydantic import BaseModel
from typing import (
    Deque, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union
)

def sha256(KEY, DATA):
    hash = hmac.new(KEY, DATA, hashlib.sha256).hexdigest()
    return hash

class GithubSigAuth(httpx.Auth):
    requires_request_body = True

    def __init__(self, token):
        self.token = bytes(token, encoding="raw_unicode_escape")

    def auth_flow(self, request):
        request.headers['X-Hub-Signature-256'] = self.sign_request(request)
        yield request

    def sign_request(self, request):
        # Create a request signature, based on `request.method`, `request.url`,
        # `request.headers`, and `request.content`.
        body=request.content
        return 'sha256=' + sha256(self.token ,body)

# sk=''
# with open('.key_gh/webhook_gpubuild', 'r') as skreader:
#     sk = skreader.readline().strip()
class WorkFlowCtrl():
    def __init__(self, baseurl, sigKey) -> None:
        self.baseURL=baseurl
        self.githubSigAuth=GithubSigAuth(sigKey)

    def sendWorkFlowNotification(self,endPoint,jsonData):
        print (self.baseURL+'/'+endPoint)
        with httpx.Client() as client:
            r=client.post(self.baseURL+'/'+endPoint, 
                auth=self.githubSigAuth, json=jsonData
            )
            rcode=r.status_code
            print(rcode,r.text)

class Workflow_End_With_SHA(BaseModel):
    name: str
    sha: str
    ref: str
    repository_uri: str
    id: int
    conclusion: Optional[str] = None
    outputs: Optional[str] = None
    gpu_yaml: Optional[str] = None    

def cmdargs():
    parser = argparse.ArgumentParser(description='A github workFlowCtrl tool',
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-u','--url',help='Set the base URL')
    parser.add_argument('-k','--sig_key', help="Set the signature key")
    parser.add_argument('-e','--end_point', help="Action")

    parser.add_argument('-n','--name', required=True, help="Workflow name")
    parser.add_argument('-s','--sha', required=True , help="Commit SHA processed in  Workflow")
    parser.add_argument('-r','--ref', default='master', help="Branch or tag processed in  Workflow")
    parser.add_argument('--repo', required=True)
    parser.add_argument('-c','--conclusion', default='')
    parser.add_argument('-o','--output', default='')
    parser.add_argument('-f','--gpu_yaml', default='gpu_test.yml')
    parser.add_argument('-t','--trigger_id', default=0, type=int, \
                        help="Workflow job id that triggered this Job")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = cmdargs()
    workFlowCtrl = WorkFlowCtrl(args.url, args.sig_key)
    workflowMessg=Workflow_End_With_SHA(name=args.name,sha=args.sha,\
        ref=args.ref,repository_uri=args.repo,conclusion=args.conclusion,\
        output=args.output, id=args.trigger_id, gpu_yaml=args.gpu_yaml)
    jsonData=json.loads(workflowMessg.json())
    workFlowCtrl.sendWorkFlowNotification(args.end_point,jsonData)

