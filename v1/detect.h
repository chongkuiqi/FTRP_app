#ifndef DETECT_H
#define DETECT_H

#endif // DETECT_H

//#include <torch/torch.h>
#undef slots
#include "torch/torch.h"
#include <torch/script.h>
#define slots Q_SLOTS


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

#define IMG_SIZE 1024
#define IMG_CHN 3


bool preprocess(const cv::Mat &img, torch::Tensor &input_tensor);
bool LoadImage(std::string file_name, cv::Mat &img);

bool postprocess(const torch::Tensor &boxes, vector<vector<Point>> &contours);
