#ifndef UTIL_H
#define UTIL_H

#endif // UTIL_H

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
#include <QString>

#define IMG_SIZE 1024
#define IMG_CHN 3


bool preprocess(cv::Mat &img, at::Tensor &input_tensor);
bool LoadImage(std::string file_name, cv::Mat &img);

bool xywhtheta2xywh4points(const at::Tensor &boxes, std::vector<std::vector<cv::Point>> &contours);


bool LoadBoxes(QString &gt_path, std::vector<std::vector<cv::Point>> &contours);
