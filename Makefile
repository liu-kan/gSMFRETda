ifdef debug
	CFLAGS = -std=c++14 -g `pkg-config --cflags hdf5`
else
	CFLAGS = -std=c++14 -O2 `pkg-config --cflags hdf5`
endif
CXXFLAGS =	$(CFLAGS)
HEADER = -I3rdparty/HighFive/include -I3rdparty/cxxopts/include -I/opt/cuda/include \
	-I3rdparty/eigen -I/usr/local/cuda-10.1/include
#BOOSTHEADER = -I 3rdparty/boost/histogram/include -I 3rdparty/boost/core/include/ -I 3rdparty/boost/iterator/include/ -I 3rdparty/boost_1_69_0/
BOOSTHEADER = -I 3rdparty/boost_1_70_0/
LIBS = `pkg-config --libs hdf5` -L/opt/cuda/lib64 -L/usr/local/cuda-10.1/lib64 -lhdf5 -lcudart  -lcurand -lnanomsg -lprotobuf -pthread
OUT_DIR=bin
MKDIR_P = mkdir -p
.PHONY: all
all: directories main
directories: ${OUT_DIR}
${OUT_DIR}:
	${MKDIR_P} ${OUT_DIR}
readhdf: src/loadHdf5.cpp directories
	$(CXX) $(CXXFLAGS) $(HEADER) -o bin/readhdf src/loadHdf5.cpp $(LIBS) 
main: src/main.cpp directories loadHdf5.o mc.o eigenhelper.o args.pb.o streamWorker.o tools.o
	$(CXX) $(CXXFLAGS) $(HEADER) -o bin/gSMFRETda src/main.cpp mc.o eigenhelper.o loadHdf5.o args.pb.o streamWorker.o tools.o $(LIBS) 
ifndef debug
	strip bin/gSMFRETda
endif
mc.o: src/mc.cu src/mc.hpp src/loadHdf5.hpp src/binom.cuh src/gen_rand.cuh src/cuList.cuh
	# nvcc $(CXXFLAGS) -arch=compute_30 -code=sm_30,sm_61,sm_70 --expt-relaxed-constexpr $(HEADER) -c src/mc.cu
	nvcc $(CXXFLAGS) -gencode arch=compute_30,code=sm_30\
					-gencode arch=compute_52,code=sm_52\
	 				-gencode arch=compute_61,code=sm_61\
					-gencode arch=compute_70,code=sm_70\
					-gencode arch=compute_75,code=sm_75\
					--expt-relaxed-constexpr $(HEADER) -c src/mc.cu
					 
loadHdf5.o:	src/loadHdf5.cpp src/loadHdf5.hpp src/bitUbyte.hpp
	$(CXX) $(CXXFLAGS) $(HEADER) -c src/loadHdf5.cpp
eigenhelper.o: src/eigenhelper.cpp src/eigenhelper.hpp
	$(CXX) $(CXXFLAGS) $(HEADER) -c src/eigenhelper.cpp
tools.o: src/tools.cpp src/tools.hpp
	$(CXX) $(CXXFLAGS) $(BOOSTHEADER) -c src/tools.cpp
tools: src/tools.cpp src/tools.hpp
	$(CXX) $(CXXFLAGS) $(BOOSTHEADER) src/tools.cpp -o bin/tools		
args.pb.o: protobuf/args.proto 
	protoc --cpp_out=src protobuf/args.proto
	protoc --python_out=serv_py protobuf/args.proto
	$(CXX) $(CXXFLAGS) $(HEADER) -Isrc -c src/protobuf/args.pb.cc 
clean:
	rm *.o
streamWorker.o: src/streamWorker.cpp src/streamWorker.hpp
	$(CXX) $(CXXFLAGS) $(HEADER) $(BOOSTHEADER) -c src/streamWorker.cpp
