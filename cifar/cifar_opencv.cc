#include <iostream>
#include <opencv2/core.hpp>
#include "opencv2/dnn.hpp"
#include <sys/time.h>
using namespace cv;
using namespace cv::dnn;
using namespace std;

unsigned long long getms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec*1.0e+3 + tv.tv_usec/1000;
}

int main(int argc, char *argv[]) {

    Net net = readNetFromTensorflow("cifar_frozen_model.pb");
    std::vector<String> names = net.getLayerNames();
    for(auto iter = names.begin(); iter != names.end(); ++iter) {
        cout << *iter << endl;
    }

    Mat im(Size(32, 32), CV_32FC3);
    Mat inputBlob = blobFromImage(im);
    cout <<inputBlob.size << endl;
    net.setInput(inputBlob);

    long long s = getms();    
    Mat out = net.forward("softmax");
    printf("forward...%lld\n", getms() - s);

    float* data = out.ptr<float>(0);

  return 0;
}
