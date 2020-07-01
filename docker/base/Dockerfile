FROM nvidia/cuda
ADD sources.list /etc/apt/
RUN apt-get update 
RUN apt-get install -y git libhdf5-dev libprotobuf-dev apt-transport-https \ 
    ca-certificates gnupg software-properties-common wget sudo \ 
    libboost-dev libboost-serialization-dev libboost-system-dev libboost-test-dev
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt-get install -y kitware-archive-keyring
RUN apt-key --keyring /etc/apt/trusted.gpg del C1F34CDD40CD72DA
RUN apt-get install -y cmake protobuf-compiler libnanomsg-dev
RUN rm -rf /var/lib/apt/lists/*