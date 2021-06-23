#include <stdio.h>
#include <string.h>
// #include <net/if.h>
// #include <sys/ioctl.h>
#include <stdlib.h>
#include <sys/types.h>
#ifdef __unix__
    #include <unistd.h>
#elif _WIN32
    #include <process.h>
    #define getpid _getpid
#endif
#include "tools.hpp"
#include <iostream>
#include <sstream> 
#include <thread> 
#include <iomanip>
#include <cstddef>
#include <cassert>
#include <cstdio>
#include <ctime>
#include <memory>
#include <random>


/*
int getC6MacAddress(unsigned char *cMacAddr,char *pIface,int ifIdx=0)
{
    int nSD;                        // Socket descriptor
    struct ifreq sIfReq;            // Interface request
    struct if_nameindex *pIfList;   // Ptr to interface name index
    struct if_nameindex *pListSave; // Ptr to interface name index
    //
    // Initialize this function
    //
    pIfList = (struct if_nameindex *)NULL;
    pListSave = (struct if_nameindex *)NULL;
    int counteth=0;
#ifndef SIOCGIFADDR
    // The kernel does not support the required ioctls
    return (0);
#endif
    //
    // Create a socket that we can use for all of our ioctls
    //
    nSD = socket(PF_INET, SOCK_STREAM, 0);
    if (nSD < 0)
    {
        // Socket creation failed, this is a fatal error
        printf("File %s: line %d: Socket failed\n", __FILE__, __LINE__);
        return (0);
    }
    //
    // Obtain a list of dynamically allocated structures
    //
    pIfList = pListSave = if_nameindex();
    // printf("pIfList:%s\n",pIfList);
    if(pIface==NULL && ifIdx==0){
        // printf("pIface:%s\n",pIface);
        for (pIfList; *(char *)pIfList != 0; pIfList++){
            strncpy(sIfReq.ifr_name, pIfList->if_name, IF_NAMESIZE);
            if (ioctl(nSD, SIOCGIFFLAGS, &sIfReq) == 0)
                if (sIfReq.ifr_flags & IFF_LOOPBACK){
                    // printf("get lo\n");
                    continue;
                }
            counteth++;
        }    
    }
    if(pIface==NULL && ifIdx!=0){
        int idx=0;
        for (pIfList; *(char *)pIfList != 0; pIfList++){
            strncpy(sIfReq.ifr_name, pIfList->if_name, IF_NAMESIZE);
            if (ioctl(nSD, SIOCGIFFLAGS, &sIfReq) == 0)
                if (sIfReq.ifr_flags & IFF_LOOPBACK){
                    // printf("get lo\n");
                    continue;
                }
            if(ifIdx==idx++){
                if (ioctl(nSD, SIOCGIFHWADDR, &sIfReq) != 0)
                {
                    // We failed to get the MAC address for the interface
                    printf("File %s: line %d: Ioctl failed\n", __FILE__, __LINE__);
                    return (0);
                }
                memmove((void *)cMacAddr, (void *)&sIfReq.ifr_ifru.ifru_hwaddr.sa_data[0], 6);
            }
            
        }    
    }    
    //
    // Walk thru the array returned and query for each interface's
    // address
    //
    else{
        for (pIfList; *(char *)pIfList != 0; pIfList++)
        {
            //
            // Determine if we are processing the interface that we
            // are interested in
            //
            if (strcmp(pIfList->if_name, pIface))
                // Nope, check the next one in the list
                continue;
            strncpy(sIfReq.ifr_name, pIfList->if_name, IF_NAMESIZE);
            //
            // Get the MAC address for this interface
            //
            if (ioctl(nSD, SIOCGIFHWADDR, &sIfReq) != 0)
            {
                // We failed to get the MAC address for the interface
                printf("File %s: line %d: Ioctl failed\n", __FILE__, __LINE__);
                return (0);
            }
            memmove((void *)cMacAddr, (void *)&sIfReq.ifr_ifru.ifru_hwaddr.sa_data[0], 6);
            break;
        }
    }
    //
    // Clean up things and return
    //
    if_freenameindex(pListSave);
    close(nSD);
    return (counteth);
}

void genuid(std::string* id){
    std::string pid=std::to_string(getpid());
    auto myid = std::this_thread::get_id();
    std::stringstream ss;
    ss << myid;
    pid+= ss.str();
    unsigned char dMacAddr[8];
    char cMacAddr[13];
    memset(dMacAddr,0, sizeof(char)*6);
    memset(cMacAddr,0, sizeof(char)*13);    
    int ethNum=getC6MacAddress(dMacAddr,NULL);
    std::string smac;
    for(int i=0;i<ethNum;i++){
        // printf("HWaddr %hhx:%hhx:%hhx:%hhx:%hhx:%hhx\n",
        //        dMacAddr[0], dMacAddr[1], dMacAddr[2],
        //        dMacAddr[3], dMacAddr[4], dMacAddr[5]);
        getC6MacAddress(dMacAddr,NULL,i);
        snprintf(cMacAddr,13,"%02X%02X%02X%02X%02X%02X",
            dMacAddr[0], dMacAddr[1], dMacAddr[2],
            dMacAddr[3], dMacAddr[4], dMacAddr[5]);
        smac+=cMacAddr;
    }
    // std::cout<<smac<<std::endl;
    *id=smac+pid;
    // std::cout<<id->size()<<std::endl;
}
*/

void genuid(std::string* id,int gid, int sid,char *gpuuid){
    std::string pid=std::to_string(getpid());
    auto tid = std::this_thread::get_id();
    std::stringstream ss,sss;
    // std::byte guid; //need c++17
    unsigned char guid;
    // char cGPUuuid[33];
    for(int i=0;i<16;i++){
        // guid=0;
        memcpy(&guid, gpuuid+i, 1);
        sss<< std::setfill ('0') << std::setw(1*2) 
       << std::hex << int(guid);
    //    std::to_integer<int>(guid); //with std::byte
    }
    ss <<sss.str() <<'-' << pid <<"-"<< tid <<"-"<<std::to_string(gid)<<"-"<<std::to_string(sid);
    *id=ss.str();
}
