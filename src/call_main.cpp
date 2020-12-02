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
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <algorithm>
#if !defined(_WIN64) && !defined(_WIN32)
    #include <signal.h>
#else
    #include <windows.h>
    #include <tlhelp32.h>
    #include <tchar.h>
    #include <Processthreadsapi.h>
    #include <Psapi.h>
    #include <Synchapi.h>
#include <strsafe.h>
#include <atlstr.h>
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

void modArg(std::string& cmdline, gengetopt_args_info& args_info){ 
    if(args_info.inputs_num<1 && !args_info.input_given ){
      std::cout<<"You need either appoint -i or add hdf5 filename in the end of cmdline!"<<std::endl;
      exit(1);
    }
    if (args_info.url_given)
        cmdline=cmdline+" -u "+args_info.url_arg;
    if (args_info.input_given)
        cmdline=cmdline+" -i "+args_info.input_arg;
    else
        cmdline=cmdline+" -i "+args_info.inputs[0];
    if (args_info.fret_hist_num_given)
        cmdline=cmdline+" -f "+args_info.fret_hist_num_orig;
    if (args_info.snum_given)
        cmdline=cmdline+" -s "+args_info.snum_orig;
    if( remove( args_info.pid_arg )!= 0 )
        perror( "Error deleting file" );
    cmdline=cmdline+" -I "+args_info.pid_arg;
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
bool pidsRunning(vector<int>& vpid){
#define BUFFER_SIZE 100
    int pidsLen=vpid.size();
    for ( int pid: vpid )
    {
        #if defined(_WIN64)|| defined(_WIN32)
            if(FindProcessId(pid)<0)
                pidsLen--;
        #else            
            if(0!=kill(pid, 0))
                pidsLen--;
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
      exit(1) ;
    std::string cmdline=gSMFRETda_path.string();
    modArg(cmdline, args_info);
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
        exit(-1);
    
    if (useAll){
        for (int igi =0;igi<nDevices;igi++){            
            #if defined(_WIN64)|| defined(_WIN32)
                std::string thecmd="START /B "+cmdline+" -g "+std::to_string(igi);
            #else
                std::string thecmd=cmdline+" -g "+std::to_string(igi)+ " &";
            #endif
            system(thecmd.c_str());
            std::this_thread::sleep_for(1s);
        }
    }
    else{
        std::vector<int> vgpuids(args_info.gpuids_arg,args_info.gpuids_arg+args_info.gpuids_given);
        std::sort(vgpuids.begin(),vgpuids.end());
        vgpuids.erase( std::unique( vgpuids.begin(), vgpuids.end() ), vgpuids.end() );
        for (int gpuid: vgpuids){
            if(gpuid<nDevices&& gpuid >=0){
                #if defined(_WIN64)|| defined(_WIN32)
                    std::string thecmd="START /B "+cmdline+" -g "+std::to_string(gpuid);
                #else
                    std::string thecmd=cmdline+" -g "+std::to_string(gpuid)+" &";
                #endif
                system(thecmd.c_str());
                std::this_thread::sleep_for(1s);
            }
        }
    }
    vector<int> vpid;
    getPid(args_info.pid_arg, vpid);
    for (int pid :vpid)
        std::cout << "pid: " <<pid<< std::endl;
    while(true){
        std::this_thread::sleep_for(1s);
        if (gSignalStatus==2){
            terminatePid(vpid);
            break;
        }
        if (!pidsRunning(vpid))
             break;
    }
    return 0;
}