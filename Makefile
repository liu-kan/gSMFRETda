ifdef debug
	CFLAGS = -std=c++11 -g `pkg-config --cflags hdf5`
else
	CFLAGS = -std=c++11 -O2 `pkg-config --cflags hdf5`
endif
CXXFLAGS =	$(CFLAGS)
HEADER = -I 3rdparty/HighFive/include -I 3rdparty/cxxopts/include -I /opt/cuda/include \
	-I 3rdparty/eigen
LIBS = -lhdf5 -L /opt/cuda/lib64 -lcudart -lcurand
OUT_DIR=bin
MKDIR_P = mkdir -p
.PHONY: directories
all: directories
directories: ${OUT_DIR}
${OUT_DIR}:
	${MKDIR_P} ${OUT_DIR}
readhdf: src/loadHdf5.cpp directories
	$(CXX) $(CXXFLAGS) $(HEADER) $(LIBS) -o bin/readhdf src/loadHdf5.cpp
main: src/main.cpp directories loadHdf5.o mc.o eigenhelper.o
	$(CXX) $(CXXFLAGS) $(HEADER) $(LIBS) -o bin/gSMFRETda src/main.cpp loadHdf5.o mc.o eigenhelper.o
mc.o: src/mc.cu src/mc.hpp
	nvcc -arch=sm_61 --expt-relaxed-constexpr $(HEADER) -c src/mc.cu
loadHdf5.o:	src/loadHdf5.cpp src/loadHdf5.hpp src/bitUbyte.hpp
	$(CXX) $(CXXFLAGS) $(HEADER) -c src/loadHdf5.cpp
eigenhelper.o: src/eigenhelper.cpp src/eigenhelper.hpp
	$(CXX) $(CXXFLAGS) $(HEADER) -c src/eigenhelper.cpp

clean:
	rm *.o