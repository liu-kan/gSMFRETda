#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/dll/runtime_symbol_info.hpp>
#include <iostream>
namespace fs = boost::filesystem;

#include "3rdparty/gengetopt/call_gSMFRETda.h"
#include "cuda_tools.hpp"
#include <cuda_runtime_api.h>
#include <chrono>
#include <thread>
#include <csignal>
#include <stdio.h>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#if !defined(_WIN64) && !defined(_WIN32)
    #include <signal.h>
    #include <sys/wait.h>
#else
    #include <windows.h>
    #include <tlhelp32.h>
    #include <tchar.h>
    #include <Processthreadsapi.h>
    #include <Psapi.h>
    #include <Synchapi.h>
// #include <strsafe.h>
// #include <atlstr.h>
    #pragma comment (lib,"Psapi.lib")
    #pragma comment (lib,"Kernel32.lib")
#endif

namespace
{
    volatile std::sig_atomic_t gSignalStatus;
}
using namespace std;

void signal_handler(int signal)
{
    gSignalStatus = signal;
}

void modArg(std::vector<string>& cmdline, gengetopt_args_info& args_info){ 
    if(args_info.inputs_num<1 && !args_info.input_given ){
      std::cout<<"You need either appoint -i or add hdf5 filename in the end of cmdline!"<<std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (args_info.url_given){
        cmdline.push_back("-u");
        cmdline.push_back(args_info.url_arg);
    }
    if (args_info.input_given){
        cmdline.push_back("-i");
        cmdline.push_back(args_info.input_arg);
    }
    else{
        cmdline.push_back("-i");
        cmdline.push_back(args_info.inputs[0]);
    }
    if (args_info.fret_hist_num_given){
        cmdline.push_back("-f");
        cmdline.push_back(args_info.fret_hist_num_orig);
    }
    if (args_info.snum_given){
        cmdline.push_back("-s");
        cmdline.push_back(args_info.snum_orig);
    }
    if( remove( args_info.pid_arg )!= 0 )
        perror( "Error deleting file" );
    cmdline.push_back("-I");
    cmdline.push_back(args_info.pid_arg);
}
void terminatePid(vector<int>& vpid){
    for ( int pid: vpid )
    {
        #if defined(_WIN64)|| defined(_WIN32)
            std::string thecmd="START /B taskkill /F /pid "+std::to_string(pid);;
            system(thecmd.c_str());
        #else            
            kill(pid, SIGKILL);
        #endif
    }
}
#if defined(_WIN64)|| defined(_WIN32)
int FindProcessId(int pid)
{
    PROCESSENTRY32 processInfo;
    processInfo.dwSize = sizeof(processInfo);
    HANDLE processesSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, NULL);
    if (processesSnapshot == INVALID_HANDLE_VALUE)
        return 0;
    Process32First(processesSnapshot, &processInfo);
    if (pid== processInfo.th32ProcessID)
    {
        CloseHandle(processesSnapshot);
        return processInfo.th32ProcessID;
    }

    while (Process32Next(processesSnapshot, &processInfo))
    {
        if (pid == processInfo.th32ProcessID)
        {
            CloseHandle(processesSnapshot);
            return processInfo.th32ProcessID;
        }
    }
    CloseHandle(processesSnapshot);
    return -1;
}
#endif
bool pidsRunning(vector<int>& vpid, std::map<int, int*>& m){
#define BUFFER_SIZE 100
    int pidsLen=vpid.size();
    printf("pidsLen %d\n",pidsLen);
    for ( int pid: vpid )
    {
        #if defined(_WIN64)|| defined(_WIN32)
            if(FindProcessId(pid)<0)
                pidsLen--;
        #else            
            // if(0!=kill(pid, 0))
            //     pidsLen--;
            pid_t w;
            w = waitpid(pid, m[pid], WNOHANG);
            if(w>0){
                printf("waitpid %d return.\n",w);
                pidsLen--;
            }
        #endif
    }
    if(pidsLen<=0)
        return false;
    else
        return true;
}
void getPid(char *pidfn, vector<int>& vpid){
    string line;
    ifstream pidfile(pidfn);
    if (pidfile) 
    {
        while (getline(pidfile, line)) {
            int pid=stoi(line);
            vpid.push_back(pid);
        }
        pidfile.close();
    }
}
int main(int argc,char** argv)
{
    using namespace std::chrono_literals;
    using namespace std::chrono;
    std::signal(SIGINT, signal_handler);
    fs::path gSMFRETda_path = boost::dll::program_location();
    //g++ src/call_main.cpp -lboost_filesystem -ldl
    gSMFRETda_path.remove_filename();
    gSMFRETda_path/="gSMFRETda";
    gengetopt_args_info args_info;
    if (cmdline_parser (argc, argv, &args_info) != 0)
      std::exit(EXIT_FAILURE);
    std::string cmdline=gSMFRETda_path.string();
    std::vector<string> cmdVect;
    cmdVect.push_back(cmdline);
    modArg(cmdVect, args_info);
    int minGid=99999;
    bool useAll=false;
    if (args_info.gpuids_given>=1){
        for (int ii=0;ii<args_info.gpuids_given;ii++){
            if (args_info.gpuids_arg[ii]<minGid)
                minGid=args_info.gpuids_arg[ii];
        }
        if (minGid==-1)
            useAll=true;
    }
    else
        useAll=true;  
    int nDevices;
    CUDA_CHECK_RETURN(cudaGetDeviceCount(&nDevices));
    if(nDevices<=0)
        std::exit(EXIT_FAILURE);
    
    std::vector<int> vgpuids;
    if (useAll)
        for (int igi =0;igi<nDevices;igi++)
            vgpuids.push_back(igi);
    else{
        vgpuids = std::vector<int>(args_info.gpuids_arg,args_info.gpuids_arg+args_info.gpuids_given);
        std::sort(vgpuids.begin(),vgpuids.end());
        vgpuids.erase( std::unique( vgpuids.begin(), vgpuids.end() ), vgpuids.end() );
    }
    int* pidret=new int[vgpuids.size()];
    std::map<int, int*> pidmap;
    for (int gpuid=0;gpuid<vgpuids.size();gpuid++){
        if(vgpuids[gpuid]<nDevices&& vgpuids[gpuid] >=0){
            #if defined(_WIN64)|| defined(_WIN32)
                std::string thecmd="START /B ";
                for (std::string argv:cmdVect)
                    thecmd=thecmd+argv+" ";
                thecmd=thecmd+" -g "+std::to_string(vgpuids[gpuid]);
                system(thecmd.c_str());                
            #else
                pid_t child_pid;
                child_pid = fork ();
                // if (child_pid != 0)
                //     /* This is the parent process.  */
                //     return child_pid;
                // else {
                if (child_pid==0){
                    /* Now execute PROGRAM, searching for it in the path.  */
                    std::vector<string> cmdv=cmdVect;
                    cmdv.push_back("-g");
                    cmdv.push_back(std::to_string(vgpuids[gpuid]));
                    const char **argvg = new const char* [cmdv.size()+1];
                    for (int j = 0;  j < cmdv.size();  ++j)
                        argvg [j] = cmdv[j].c_str();
                    argvg[cmdv.size()]=NULL;
                    execv (cmdVect[0].c_str(), (char **)argvg);
                    /* The execvp function returns only if an error occurs.  */
                    fprintf (stderr, "an error occurred in execv\n");
                    exit(EXIT_FAILURE);
                }                
            #endif
            std::this_thread::sleep_for(1s);
        }
    }    
    vector<int> vpid;
    getPid(args_info.pid_arg, vpid);
    for (int pididx=0;pididx<vpid.size();pididx++){
        std::cout << "pid: " <<vpid[pididx]<< std::endl;
        pidmap[vpid[pididx]]=pidret+pididx;
    }
    while(true){
        std::this_thread::sleep_for(1s);
        if (gSignalStatus==SIGINT){
            printf("SIGINT got\n");
            terminatePid(vpid);
            break;
        }
        if (!pidsRunning(vpid,pidmap))
            break;
    }
    int retcode=EXIT_SUCCESS;
    #if !defined(_WIN64) && !defined(_WIN32)
        for (int gpuid=0;gpuid<vgpuids.size();gpuid++){        
            if (WIFEXITED(pidret[gpuid])) {
                int tretcode=WEXITSTATUS(pidret[gpuid]);
                printf("exit code:%d\n",tretcode);
                if(tretcode!=EXIT_SUCCESS)
                    retcode=tretcode;
            } 
            else
                retcode=EXIT_FAILURE;
        }
    #endif
    delete[] pidret;
    return retcode;
}