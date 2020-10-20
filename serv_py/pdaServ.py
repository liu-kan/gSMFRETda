import signal
from multiprocessing import Process, Value
# TODO use thread instead multiprocessing
import sys, argparse,multiprocessing
from msg import paramsServ
from opt import opt_toobox
ctrlc=0

def main_function():
    stopFlag = Value('b', 0)
    dbname=''
    savefn=''
    state=2
    pick='out.pickle'
    bins=70
    maxiter=1000
    parser = argparse.ArgumentParser(description='A optimizing skeleton to provide parameters for gSMFRETda',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=
'''
Pls cite the 
work.
 
''')
    parser.add_argument('-p','--port',default='7777',help='listening port (default 7777)')
    parser.add_argument('-s','--s_n',default=3, type=int, help="states' number (default 3)")
    parser.add_argument('-i','--ind_num',default=0, type=int, help="individual number of one gen")
    args = parser.parse_args()
    # https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
    # https://www.cloudcity.io/blog/2019/02/27/things-i-wish-they-told-me-about-multiprocessing-in-python/
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    qO  = multiprocessing.Queue()
    qN = multiprocessing.Queue()
    pServ = paramsServ(args.port,args.s_n)
    q=(qO,qN)
    paramsServ_p = Process(target=pServ.run, args=(stopFlag,q))
    optBox=opt_toobox(args.s_n)
    optBox_p=Process(target=optBox.run, args=(stopFlag,q,args.ind_num))
    def exit_handler(signal_received, frame):
        global ctrlc
        ctrlc=ctrlc+1
        print("Ctrl+c press, program try to end gracefully! Press Ctrl+c twice to quit immediately.")
        stopFlag.value=1
        if ctrlc>1:
            stopFlag.value=2
            paramsServ_p.terminate()
            optBox_p.terminate()
            # time.sleep(1)
            sys.exit()
        # time.sleep(1)
    paramsServ_p.daemon = True
    optBox_p.daemon = True
    optBox_p.start()
    paramsServ_p.start()
    signal.signal(signal.SIGINT, original_sigint_handler)
    signal.signal(signal.SIGINT, exit_handler)
    paramsServ_p.join()
    optBox_p.join()
    # 
    # %%capture output
    # output.show()    


if __name__ == '__main__':
    main_function()
