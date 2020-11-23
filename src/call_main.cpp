#include <boost/process/group.hpp>
#include <boost/process/spawn.hpp>
namespace bp = boost::process; 

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/dll/runtime_symbol_info.hpp>
#include <iostream>
namespace fs = boost::filesystem;

#include "3rdparty/gengetopt/call_gSMFRETda.h"

void spawn(std::string pp)
{
    bp::group g;
    bp::spawn(pp.c_str(), g);
    // bp::spawn("bar", g);
    g.wait();
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
}

int main(int argc,char** argv)
{

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
        for (int igi=0;igi<args_info.gpuids_given;igi++){
            if (args_info.gpuids_arg[igi]<minGid)
                minGid=args_info.gpuids_arg[igi];
        }
        if (minGid==-1)
            useAll=true;
    }
    
    spawn(gSMFRETda_path.string()+" -h");

    return 0;
}