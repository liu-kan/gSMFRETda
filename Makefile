ifdef debug
	CFLAGS = -std=c++11 -g `pkg-config --cflags hdf5`
else
	CFLAGS = -std=c++11 -O2 `pkg-config --cflags hdf5`
endif
CXXFLAGS =	$(CFLAGS)
HEADER = -I3rdparty/HighFive/include -I3rdparty/cxxopts/include -I/opt/cuda/include \
	-I3rdparty/eigen -I/usr/local/cuda/include -I3rdparty/cpp-base64
LIBS = -L/opt/cuda/lib64 -L/usr/local/cuda/lib64 -lhdf5 -lcudart  -lcurand -lnanomsg -lprotobuf -pthread -lPocoFoundation
OUT_DIR=bin
MKDIR_P = mkdir -p
.PHONY: directories
all: directories
directories: ${OUT_DIR}
${OUT_DIR}:
	${MKDIR_P} ${OUT_DIR}
readhdf: src/loadHdf5.cpp directories
	$(CXX) $(CXXFLAGS) $(HEADER) -o bin/readhdf src/loadHdf5.cpp $(LIBS) 
main: src/main.cpp directories loadHdf5.o mc.o eigenhelper.o args.pb.o base64.o tools.o
	$(CXX) $(CXXFLAGS) $(HEADER) -o bin/gSMFRETda src/main.cpp mc.o eigenhelper.o loadHdf5.o args.pb.o base64.o tools.o $(LIBS) 
mc.o: src/mc.cu src/mc.hpp src/loadHdf5.hpp src/binom.cuh src/gen_rand.cuh src/cuList.cuh
	nvcc $(CXXFLAGS) -arch=sm_61 --expt-relaxed-constexpr $(HEADER) -c src/mc.cu
loadHdf5.o:	src/loadHdf5.cpp src/loadHdf5.hpp src/bitUbyte.hpp
	$(CXX) $(CXXFLAGS) $(HEADER) -c src/loadHdf5.cpp
eigenhelper.o: src/eigenhelper.cpp src/eigenhelper.hpp
	$(CXX) $(CXXFLAGS) $(HEADER) -c src/eigenhelper.cpp
tools.o: src/tools.cpp src/tools.hpp
	$(CXX) $(CXXFLAGS) -c src/tools.cpp	
args.pb.o: protobuf/args.proto 
	protoc --cpp_out=src protobuf/args.proto
	protoc --python_out=serv_py protobuf/args.proto
	$(CXX) $(CXXFLAGS) $(HEADER) -Isrc -c src/protobuf/args.pb.cc 
clean:
	rm *.o
base64.o: 3rdparty/cpp-base64/base64.cpp 3rdparty/cpp-base64/base64.h
	$(CXX) $(CXXFLAGS) -c 3rdparty/cpp-base64/base64.cpp
