CC = g++
CFLAGS = -g -Wall  -O2 -std=c++11
SRCS = mnist_opencv.cc
PROG = mnist_cv
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(INC) $(LIBS)

