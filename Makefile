CFLAGS = 	-std=c++11 -I 3rdparty/HighFive/include -O2
CXXFLAGS =	$(CFLAGS)
LIBS = -lhdf5
OUT_DIR=bin
MKDIR_P = mkdir -p
.PHONY: directories
all: directories
directories: ${OUT_DIR}
${OUT_DIR}:
	${MKDIR_P} ${OUT_DIR}
readhdf: src/loadHdf5.cpp directories
	$(CXX) $(CXXFLAGS) $(LIBS) -o bin/readhdf src/loadHdf5.cpp