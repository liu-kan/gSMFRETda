ifdef debug
	CFLAGS = -std=c++11 -g `pkg-config --cflags hdf5`
else
	CFLAGS = -std=c++11 -O2 `pkg-config --cflags hdf5`
endif
CXXFLAGS =	$(CFLAGS)
HEADER = -I3rdparty/HighFive/include -I3rdparty/cxxopts/include -I/opt/cuda/include \
	-I3rdparty/eigen -I/usr/local/cuda/include
LIBS = `pkg-config --libs-only-L hdf5` -L/opt/cuda/lib64 -L/usr/local/cuda/lib64 -lhdf5 -lcudart  -lcurand
OUT_DIR=bin
MKDIR_P = mkdir -p
.PHONY: directories
all: directories
directories: ${OUT_DIR}
${OUT_DIR}:
	${MKDIR_P} ${OUT_DIR}
readhdf: src/loadHdf5.cpp directories
	$(CXX) $(CXXFLAGS) $(HEADER) -o bin/readhdf src/loadHdf5.cpp $(LIBS) 
main: src/main.cpp directories loadHdf5.o mc.o eigenhelper.o
	$(CXX) $(CXXFLAGS) $(HEADER) -o bin/gSMFRETda src/main.cpp mc.o eigenhelper.o loadHdf5.o $(LIBS) 
mc.o: src/mc.cu src/mc.hpp src/loadHdf5.hpp src/binom.cuh src/gen_rand.cuh src/cuList.cuh
	nvcc $(CXXFLAGS) -arch=sm_61 --expt-relaxed-constexpr $(HEADER) -c src/mc.cu
loadHdf5.o:	src/loadHdf5.cpp src/loadHdf5.hpp src/bitUbyte.hpp
	$(CXX) $(CXXFLAGS) $(HEADER) -c src/loadHdf5.cpp
eigenhelper.o: src/eigenhelper.cpp src/eigenhelper.hpp
	$(CXX) $(CXXFLAGS) $(HEADER) -c src/eigenhelper.cpp

clean:
	rm *.o