CC = g++
CFLAGS = -g -Wall  -O2 -std=c++11
SRCS = cifar_opencv.cc
PROG = cifar_cv
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(INC) $(LIBS)

