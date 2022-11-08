#ifndef UTIL_H
#define UTIL_H

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
#include <QImage>

// 泛型编程，存储各个特征的状态、特征相似度
template <typename T>
struct struct_MS
{
    T opt;
    T IR;
    T SAR;
};

// 模仿cv::RotatedRect，构建一个结构体
struct RBox
{
    // 旋转框中心点坐标
    cv::Point center;
    // 长边和短边，长边为width，短边为height
    cv::Size  size;
    // 角度，逆时针转动为正，[-45°~135°]
    float angle;
};



#define IMG_SIZE 1024
#define IMG_CHN 3



void preprocess(cv::Mat &img, at::Tensor &input_tensor);

bool LoadImage(std::string file_name, cv::Mat &img);

void points2rbox(const std::vector<cv::Point> & contour, RBox &rbox);
void rbox2points(std::vector<cv::Point> & contour, const RBox &rbox);

//void points2rbox(const std::vector<cv::Point> & contour, cv::RotatedRect &rbox);
//void rbox2points(std::vector<cv::Point> & contour, const cv::RotatedRect &rbox);
void points2xywhtheta(const std::vector<cv::Point> & contour, at::Tensor &rbox);
void xywhtheta2points(std::vector<cv::Point> & contour, const at::Tensor &rbox);
void xywhtheta2points(const at::Tensor &boxes, std::vector<std::vector<cv::Point>> &contours);


void LoadBoxes(QString &gt_path, std::vector<std::vector<cv::Point>> &contours);


// cv::Mat与QImage之间的转换
QImage MatToImage(const cv::Mat &m);  //Mat转Image
QPixmap MatToPixmap(const cv::Mat &m);
cv::Mat ImageToMat(const QImage &img,bool inCloneImageData);  //Image转Mat



#endif // UTIL_H





