#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>

#include "roi_align_rotated_cuda.h"

using namespace std;
using namespace cv;

#define IMG_SIZE 1024
#define IMG_CHN 3

float img_normalize_mean[3] = {123.675, 116.28, 103.53};
float img_normalize_std[3] = {58.395, 57.12, 57.375};

bool LoadImage(std::string file_name, cv::Mat &img)
{
  img = cv::imread(file_name); // CV_8UC3
  if (img.empty() || !img.data)
    {
      cout << "failed to read img " << endl;
      return false;
    }
  return true;
}

// 
bool preprocess(const cv::Mat &img, torch::Tensor &input_tensor)
{

  input_tensor = torch::from_blob(
    img.data, {1, IMG_SIZE, IMG_SIZE, IMG_CHN});
  input_tensor = input_tensor.permute({0, 3, 1, 2});

  input_tensor[0][0] = input_tensor[0][0].sub_(img_normalize_mean[0]).div_(img_normalize_std[0]);
  input_tensor[0][1] = input_tensor[0][1].sub_(img_normalize_mean[1]).div_(img_normalize_std[1]);
  input_tensor[0][2] = input_tensor[0][2].sub_(img_normalize_mean[2]).div_(img_normalize_std[2]);
}

int main(int argc, const char *argv[]) 
{

  torch::Device device(torch::kCUDA, 0);
  // torch::Device device(torch::kCPU);


  if (argc != 3)
  {
    std::cerr << "Usage:classifier <path-to-exported-script-module>  <path-to-lable-file> " << std::endl;
    return -1;
  }

  std::string model_path = argv[1];
  std::string img_path = argv[2];

  cv::Mat img = cv::imread(img_path); // CV_8UC3
  cv:Mat img2;
  cv::cvtColor(img, img2, cv::COLOR_BGR2RGB);
  // // scale image to fit
  cv::Size scale(IMG_SIZE, IMG_SIZE);
  cv::resize(img2, img2, scale);
  
  // convert [unsigned int] to [float]
  img2.convertTo(img2, CV_32FC3);


  // cout << img.at<float>(0,0,0) << endl;
  //read image tensor
  torch::Tensor input_tensor;
  preprocess(img2, input_tensor);
  // to GPU
  input_tensor = input_tensor.to(device);

  // cout << img.itemsize() << endl;
  cout << "input img size:" << input_tensor.sizes() << endl;



  torch::jit::script::Module model;
  model = torch::jit::load(model_path, device);
  // module = torch::jit::load("/home/ckq/software/example/FTRP_software/model_no_align_script.pt", deviceCUDA);
  model.eval();

  torch::NoGradGuard no_grad;

  cout << "开始推理..." << endl;
  auto output = model.forward({input_tensor}).toTuple();
  // auto output = model.forward({input_tensor});
  // cout << output << endl;
  cout << "结束！" << endl;
  // auto output = module.forward({input_tensor}).toList();
  // exit();

  torch::Tensor feat = output->elements()[0].toTensor();
  cout << feat.is_contiguous() << endl;
  feat = feat.contiguous();
  cout << feat.is_contiguous() << endl;

  int out_h=7, out_w=7, sample_num=2;
  float spatial_scale = 1/8;

  torch::Tensor rois = torch::ones({2,6}).to(device);
  
  torch::Tensor outs = torch::ones({2,256,7,7}).to(device);
  int roi_align_rotated_forward_cuda(at::Tensor features, at::Tensor rois,
                           int pooled_height, int pooled_width,
                           float spatial_scale, int sample_num,
                           at::Tensor output);
  int a = roi_align_rotated_forward_cuda(feat, rois, out_h, out_w, spatial_scale, sample_num, outs);

  cout << outs << endl;
  // // only one img
  // auto b = output->elements()[0];
  
  // // cout << "检测结果后处理" << endl;

  // torch::Tensor boxes = b.toTuple()->elements()[0].toTensor();
  // cout << "5参数:" << boxes << endl;
  // float x = (boxes[0][0]).item().toFloat();
  // float y = (boxes[0][1]).item().toFloat();
  // float w = (boxes[0][2]).item().toFloat();
  // float h = (boxes[0][3]).item().toFloat();
  // float theta = (boxes[0][4]).item().toFloat() * 180 / M_PI;
  // // if (theta < 0) 
  // // {
  // //   theta = -theta;
  // // }
  // // else
  // // {
  // //   theta = 180 - theta;
  // // }

  

  // RotatedRect rRect = RotatedRect(Point2f(x,y), Size2f(w,h), theta);
  // Point2f vertices[4];      //定义4个点的数组
  // rRect.points(vertices);   //将四个点存储到vertices数组中

  // // drawContours函数需要的点是Point类型而不是Point2f类型
  // vector<vector<Point>> contours;
  // vector<Point> contour;
  // for (int i=0; i<4; i++)
  // {
  //   contour.emplace_back(Point(vertices[i]));
  // }
  // contours.push_back(contour);

  // // // // torch::Tensor labels = b.toTuple()->elements()[1].toTensor();
  // // torch::Tensor points = b.toTuple()->elements()[2].toTensor();
  
  

  // // cout << "准备读取tensor内容" << endl;
  // // void *ptr = points.data_ptr();

  // // // cout << "创建指针" << endl;
  // // // cout << "指针内容：" << *((float*)ptr+2) << endl;
  // // for(int i=0; i<8; i=i+2)
  // // {
  // //   Point p((int)*((float*)ptr+i), (int)*((float*)ptr+i+1));
  // //   // Point2f p(*((float*)ptr+i), *((float*)ptr+i+1));
  // //   // contour.push_back(p);  
  // //   contours[0][i] = p;
  // // }

  // // contours.push_back(contour);


  // // findContours(img2, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
  // // cout << contours[0] << endl;
  // // cout << "轮廓个数:" << contours[0].size() << endl;
  

  // cout << "开始画图" << endl;

  // // // convert [unsigned int] to [float]
  // // img.convertTo(img, CV_8UC3);
  // // cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
  // int contoursIds = -1;
  // const Scalar color = Scalar(0,0,255);
  // int thickness = 3;
  // drawContours(img, contours, contoursIds, color, thickness);
  // namedWindow( "image", 1 );
  // imshow( "image", img );
  // waitKey(0);

}




