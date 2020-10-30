FROM liukan/gsmfretda
WORKDIR /build
COPY filelist.md5 dlfiles.sh ./
RUN rm -rf gSMFRETda; git clone --recurse-submodules -j8 https://github.com/liu-kan/gSMFRETda.git; mkdir -p gSMFRETda_build; chmod +x ./dlfiles.sh; ./dlfiles.sh
WORKDIR /build/gSMFRETda_build
RUN cmake ../gSMFRETda && make -j