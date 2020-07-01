#include <stdio.h>
#include <string.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include "tools.hpp"
#include <iostream>
#include <sstream> 
#include <thread> 


#include <cassert>
#include <cstdio>
#include <ctime>
#include <memory>
#include <random>




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
        // printf("HWaddr %2X:%2x:%2x:%2x:%2x:%2X\n",
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


std::unique_ptr<double[]> random_array(unsigned n, int type) {
  std::unique_ptr<double[]> r(new double[n]);
  std::default_random_engine gen(1);
  if (type) { // type == 1
    std::normal_distribution<> d(0.5, 0.3);
    for (unsigned i = 0; i < n; ++i) r[i] = d(gen);
  } else { // type == 0
    std::uniform_real_distribution<> d(0.0, 1.0);
    for (unsigned i = 0; i < n; ++i) r[i] = d(gen);
  }
  return r;
}
#include <boost/format.hpp>
template <typename Tag, typename Storage>
double compare_1d(unsigned n, int distrib) {
  auto r = random_array(n, distrib);
  auto h = make_s(Tag(), Storage(), reg(100, 0, 1));
  auto t = clock();
  double timesp=(double(clock()) - t) / CLOCKS_PER_SEC / n * 1e9;
  for (auto it = r.get(), end = r.get() + n; it != end;) h(*it++);
      for (auto x : indexed(h)) {
      std::cout << boost::format("bin %3i [%4.4f, %4.4f): %i\n")
        % x.index() % x.bin().lower() % x.bin().upper() % *x;
    }

  return timesp;
}
double baseline(unsigned n) {
  auto r = random_array(n, 0);
  auto t = clock();
  for (auto it = r.get(), end = r.get() + n; it != end;) {
    volatile auto x = *it++;
    (void)(x);
  }
  return (double(clock()) - t) / CLOCKS_PER_SEC / n * 1e9;
}
auto mkhist(std::vector<float>* SgDivSr,int binnum,float lv,float uv){
    auto h = make_s(static_tag(), std::vector<float>(), reg(binnum, lv, uv));
    for (auto it = SgDivSr->begin(), end = SgDivSr->end(); it != end;) 
        h(*it++);
    // auto h = make_histogram(
    //   axis::regular<>(binnum, 0.0, 1.0, "x")
    // );    
    // std::for_each(SgDivSr->begin(), SgDivSr->end(), std::ref(h));
    return h;
}



int _main(){    
    std::string id;
    genuid(&id);
    std::cout<<id<<std::endl;

    const unsigned nfill = 2000;
    using SStore = std::vector<int>;
    using SFStore = std::vector<float>;
    using DStore = unlimited_storage<>;
    printf("baseline %.1f\n", baseline(nfill));
    for (int itype = 0; itype < 2; ++itype) {
        const char* d = itype == 0 ? "uniform" : "normal ";
        printf("1D-boost:SS-%s %5.1f\n", d, compare_1d<static_tag, SStore>(nfill, itype));
        printf("1D-boost:SD-%s %5.1f\n", d, compare_1d<static_tag, DStore>(nfill, itype));
        printf("1D-boost:DS-%s %5.1f\n", d, compare_1d<dynamic_tag, SStore>(nfill, itype));
        printf("1D-boost:DD-%s %5.1f\n", d, compare_1d<dynamic_tag, DStore>(nfill, itype));

    }
    std::vector<float> s={-34.4,0.435,5.3,.67,.45,.72,.56487,.678,.89,.432};
    auto h=mkhist(&s,3,0,1);//<static_tag,SFStore>
    
        for (auto x : indexed(h)) {
        std::cout << boost::format("bin %3i [%4.4f, %4.4f): %i\n")
        % x.index() % x.bin().lower() % x.bin().upper() % *x;
    }
exit(0);
}