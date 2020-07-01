#!/bin/bash
wget http://172.16.0.2:2020/boost_1_70_0.tar.bz2 -O /build/gSMFRETda/3rdparty/boost_1_70_0.tar.bz2
if md5sum -c filelist.md5; then
    echo "===Use localhost boost_1_70_0.tar.bz2.==="
else
    rm gSMFRETda/3rdparty/boost_1_70_0.tar.bz2
    echo "====Download from boost website later.===="
fi