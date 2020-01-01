#include <iostream>
#include <opencv2/core.hpp>
#include "opencv2/dnn.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main(int argc, char *argv[]) {

    Net net = readNetFromTensorflow("mnist_frozen_model.pb");
    std::vector<String> names = net.getLayerNames();
    for(auto iter = names.begin(); iter != names.end(); ++iter) {
        cout << *iter << endl;
    }


    Mat im(Size(28, 28), CV_32FC1);
    Mat inputBlob = blobFromImage(im);
    net.setInput(inputBlob);
    Mat out = net.forward("softmax");
    float* data = out.ptr<float>(0);
    cout<<out.size<<endl;
    cout<<data[0]<<endl;
    cout<<data[1]<<endl;

  return 0;
}
