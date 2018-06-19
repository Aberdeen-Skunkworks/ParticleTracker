.DEFAULT_GOAL:= all
CXXFLAGS:= --std=c++11 -Wall $(shell pkg-config python3 --cflags) -O3
LINKFLAGS:= $(shell pkg-config python3 --libs)
PYTHON_VER:= $(shell pkg-config python3 --modversion)
LINKFLAGS += -lboost_python-py$(subst .,,${PYTHON_VER})

ImageProc.so: ImageProc.pyx
	cython ImageProc.pyx
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o ImageProc.so ImageProc.c

all: ImageProc.so
