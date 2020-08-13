from __future__ import print_function
from protobuf import args_pb2
from nanomsg import Socket, REP
import time,base64
import signal
from multiprocessing import Process, Value
import sys,getopt

ctrlc=0

def main_function():
    stopFlag = Value('b', 0)
    dbname=''
    savefn=''
    state=2
    pick='out.pickle'
    bins=70
    maxiter=1000
    port='7777'
    try:  
        opts, args = getopt.getopt(sys.argv[1:], "l:i:s:o:b:m:p:", ["log=", "dat=","state=","pickle=","bin=","maxiter=","port="])  
        for o, v in opts: 
            if o in ("-l", "--log"):
                savefn = v
            if o in ("-b", "--bin"):
                bins = int(v.strip())                
            if o in ("-p", "--port"):
                port = v
            if o in ("-m", "--maxiter"):
                maxiter = int(v.strip())                                
            if o in ("-i", "--dat"):
                dbname=v
            if o in ("-o", "--pickle"):
                pick=v                
            if o in ("-s", "--state"):
                state = int(v.strip())
                # print(state)

    except getopt.GetoptError:  
        print("getopt error!")    
        # usage()    
        sys.exit(1)

    # https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    paramsServ_p = Process(target=paramsServ, args=(port,stopFlag))

    def exit_handler(signal_received, frame):
        global ctrlc
        ctrlc=ctrlc+1
        print("Ctrl+c press, program try to end gracefully! Press Ctrl+c twice to quit immediately.")
        stopFlag.value=1
        if ctrlc>1:
            stopFlag.value=2
            paramsServ_p.terminate()
            # time.sleep(1)
            sys.exit()
        # time.sleep(1)

    paramsServ_p.start()
    signal.signal(signal.SIGINT, original_sigint_handler)
    signal.signal(signal.SIGINT, exit_handler)
    paramsServ_p.join()
    # 
    # %%capture output
    # output.show()    

class gpuClient:
    runTime=99999.9
    timeStamp=-1
    def __init__(self):
        self.timeStamp=time.perf_counter()
    def updateRunTime(self):
        self.runTime=time.perf_counter()-self.timeStamp
        self.timeStamp=time.perf_counter()
    def updateTimeStamp(self):
        self.timeStamp=time.perf_counter()


def paramsServ(port,stopflag):
    print('tcp://*:'+port)
    s1 = Socket(REP)
    s1.bind('tcp://*:'+port)    
    pb_n=args_pb2.p_n()
    s_n=3
    ps_n=s_n*(s_n+1)
    pb_n.s_n=s_n
    print("nanoMesg up")
    time.sleep(0.5)
    running=True
    pb_sid=args_pb2.p_sid()
    pb_sid.sid=0
    clients=dict()
    clients_lastTimeS=dict()
    while(running):
        # print("loop start")
        if stopflag.value>1:
            running=False
            break
        recvstr=s1.recv()
        # print(recvstr[0],ord('c'),ord('p'))
        if recvstr[0]==ord('c'):
            pb_cap=args_pb2.p_cap()
            pb_cap.ParseFromString(recvstr[1:])
            # print("pb_cap.idx",pb_cap.idx)
            s1.send(pb_n.SerializeToString())
        elif recvstr[0]==ord('p'):
            pb_ga=args_pb2.p_ga()
            pb_ga.start=0
            pb_ga.stop=-1
            # print(base64.b64decode(recvstr[1:]))
            # pb_ga.idx=base64.b64decode(recvstr[1:])
            pb_ga.idx=recvstr[1:]
            # print("pb_ga.idx",pb_ga.idx)
            for i in range(s_n):
                pb_ga.params.append((i+1)*0.27)
            for i in range(s_n*s_n-s_n):
                pb_ga.params.append((i+1)*40)
            for i in range(s_n):
                pb_ga.params.append(30)                
            s1.send(pb_ga.SerializeToString())
            # clients_lastTimeS[pb_ga.idx]=time.perf_counter()
            # print("p: ",clients_lastTimeS[pb_ga.idx])
            if pb_ga.idx not in clients:
                clients[pb_ga.idx]=gpuClient()
        elif recvstr[0]==ord('r'):
            res=args_pb2.res()
            res.ParseFromString(recvstr[1:])
            if res.idx in clients:
                # print("r: ",clients_lastTimeS[res.idx])
                clients[res.idx].updateRunTime()
            print("res chi2",res.e)
            if stopflag.value >=1:
                pb_sid.sid=-1
                
                print("time spend: ",clients.pop(res.idx).runTime)
                if len(clients.keys())<=0:
                    running=False
                print("stopflag: ",stopflag.value)
                # s1.close()
            else:
                print("stopflag 0: ",stopflag.value)
                pass
            s1.send(pb_sid.SerializeToString())
    s1.close()
if __name__ == '__main__':
    main_function()
