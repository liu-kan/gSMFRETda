ifdef debug
	CFLAGS = -std=c++11 -g
else
	CFLAGS = -std=c++11 -O2
endif
CXXFLAGS =	$(CFLAGS)
HEADER = -I 3rdparty/HighFive/include -I 3rdparty/cxxopts/include
LIBS = -lhdf5
OUT_DIR=bin
MKDIR_P = mkdir -p
.PHONY: directories
all: directories
directories: ${OUT_DIR}
${OUT_DIR}:
	${MKDIR_P} ${OUT_DIR}
readhdf: src/loadHdf5.cpp directories
	$(CXX) $(CXXFLAGS) $(HEADER) $(LIBS) -o bin/readhdf src/loadHdf5.cpp
main: src/main.cpp directories loadHdf5.o
	$(CXX) $(CXXFLAGS) $(HEADER) $(LIBS) -o bin/gSMFRETda src/main.cpp loadHdf5.o

loadHdf5.o:	src/loadHdf5.cpp src/loadHdf5.hpp
	$(CXX) $(CXXFLAGS) $(HEADER) -c src/loadHdf5.cpp