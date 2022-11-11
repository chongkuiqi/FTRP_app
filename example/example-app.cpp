#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>

#include "roi_align_rotated_cpu.h"

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
void preprocess(cv::Mat &img, torch::Tensor &input_tensor)
{

  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  // // // scale image to fit
  // cv::Size scale(IMG_SIZE, IMG_SIZE);
  // cv::resize(img, img, scale);
  // convert [unsigned int] to [float]
  img.convertTo(img, CV_32FC3);

  input_tensor = torch::from_blob(
    img.data, {1, IMG_SIZE, IMG_SIZE, IMG_CHN});
  input_tensor = input_tensor.permute({0, 3, 1, 2});

  input_tensor[0][0] = input_tensor[0][0].sub_(img_normalize_mean[0]).div_(img_normalize_std[0]);
  input_tensor[0][1] = input_tensor[0][1].sub_(img_normalize_mean[1]).div_(img_normalize_std[1]);
  input_tensor[0][2] = input_tensor[0][2].sub_(img_normalize_mean[2]).div_(img_normalize_std[2]);

  input_tensor = input_tensor.contiguous();
}

int map_roi_levels(float roi_scale, int num_levels, int finest_scale=56)
{

    int level_id = floor(log2(roi_scale / finest_scale + 1e-6));
    level_id = (level_id<0) ? 0 : level_id;
    level_id = (level_id>(num_levels-1)) ? (num_levels-1) : level_id;

    return level_id;
}

void get_fe_deep(cv::Mat &img, const at::Tensor &rbox, at::Tensor &roi_fe_deep)
{
    std::string model_path = "/home/ckq/MyDocument/QtCode/ftrp_software/example/FTRP_software/exp361_no_align_cuda_script.pt";

    torch::Device device(torch::kCPU);

    // 提取第一个区域的特征
    at::Tensor input_tensor;
    preprocess(img, input_tensor);
    input_tensor = input_tensor.to(device);
//     cout << "input img size:" << input_tensor.sizes() << endl;

    torch::NoGradGuard no_grad;
    torch::jit::script::Module model;
    model = torch::jit::load(model_path, device);

    c10::intrusive_ptr<at::ivalue::Tuple> output;
    output = model.forward({input_tensor}).toTuple();



    int out_h=7, out_w=7, sample_num=2;
    int finest_scale=56; //Scale threshold of mapping to level 0.
    int featmap_strides[3] = {8,16,32};

    // 获得roi所在的特征层级
    int num_levels=3;
    float roi_1_scale = (rbox[0][3] * rbox[0][4]).item().toFloat();
    int feat_level_id = map_roi_levels(roi_1_scale, num_levels, finest_scale);
    at::Tensor feat = output->elements()[feat_level_id].toTensor();

    float spatial_scale = 1/featmap_strides[feat_level_id];
    int chn = feat.sizes()[1];
    int num_rboxes = rbox.sizes()[0];

    roi_fe_deep = feat.new_zeros({num_rboxes, chn, out_h, out_w});
    roi_align_rotated_forward_cpu(feat, rbox, out_h, out_w, spatial_scale, sample_num, roi_fe_deep);
    roi_fe_deep = roi_fe_deep.reshape(-1);

}


void test()
{
  for (int i=0; i<10; i++)
  {
    const int num = i;
    std::cout << "第几次：" << num << std::endl;
  }
}

int main(int argc, const char *argv[]) 
{

  // torch::Device device(torch::kCUDA, 0);
  torch::Device device(torch::kCPU);


  if (argc != 3)
  {
    std::cerr << "Usage:classifier <path-to-exported-script-module>  <path-to-lable-file> " << std::endl;
    return -1;
  }

  std::string model_path = argv[1];
  std::string img_path = argv[2];

  cv::Mat img = cv::imread(img_path); // CV_8UC3
  cv:Mat img2=img.clone();
  
  // cout << img.at<float>(0,0,0) << endl;
  //read image tensor
  torch::Tensor input_tensor;
  preprocess(img2, input_tensor);
  input_tensor = input_tensor.to(device);
  // cout << img.itemsize() << endl;
  cout << "input img size:" << input_tensor.sizes() << endl;


  torch::jit::script::Module model;
  model = torch::jit::load(model_path, device);
  // module = torch::jit::load("/home/ckq/software/example/FTRP_software/model_no_align_script.pt", deviceCUDA);
  model.eval();

  torch::NoGradGuard no_grad;

  cout << "开始推理..." << endl;
  std::cout << "例子" << std::endl;
  std::cout << input_tensor.sizes() << std::endl;


  c10::intrusive_ptr<at::ivalue::Tuple> output;
  output = model.forward({input_tensor}).toTuple();
  cout << "结束！" << endl;
  // auto output = module.forward({input_tensor}).toList();
  // exit();



  int out_h=7, out_w=7, sample_num=2;
  int finest_scale=56; //Scale threshold of mapping to level 0.
  int featmap_strides[3] = {8,16,32};
  int num_levels=3;

  // shape [N,6(batch_id, x,y,w,h,theta)]
  torch::Tensor rbox_1 = torch::tensor(
    {0.0000 , 443.0000  ,614.0000  ,166.0000  , 19.0000   ,-0.3122}
  ).to(device).reshape({-1,6});

  float roi_1_scale = (rbox_1[0][3] * rbox_1[0][4]).item().toFloat();
  int feat_level_id = map_roi_levels(roi_1_scale, num_levels, finest_scale);
  // feat_level_id = 1;
  at::Tensor feat_1 = output->elements()[feat_level_id].toTensor();
  std::cout << "特征层级:" << feat_level_id << std::endl;
  float spatial_scale = 1./featmap_strides[feat_level_id];
  int chn = feat_1.sizes()[1];
  int num_rboxes = rbox_1.sizes()[0];

  at::Tensor roi_fe_1 = feat_1.new_zeros({num_rboxes, chn, out_h, out_w});
  // std::cout << feat_1 << std::endl;
  roi_align_rotated_forward_cpu(feat_1, rbox_1, out_h, out_w, spatial_scale, sample_num, roi_fe_1);
  roi_fe_1 = roi_fe_1.reshape(-1);


  torch::Tensor rbox_2 = torch::tensor(
    {0.0000 , 988.0000 , 190.0000  , 50.0000 ,   9.0000  ,  1.3306}
    // {0.0000 , 988.0000 , 190.0000  , 50.0000 ,   9.0000  ,  -0.3122}
  ).to(device).reshape({-1,6});

  float roi_2_scale = (rbox_2[0][3] * rbox_2[0][4]).item().toFloat();
  feat_level_id = map_roi_levels(roi_2_scale, num_levels, finest_scale);
  // feat_level_id = 1;
  at::Tensor feat_2 = output->elements()[feat_level_id].toTensor();
  std::cout << "特征层级:" << feat_level_id << std::endl;
  spatial_scale = 1./featmap_strides[feat_level_id];
  chn = feat_2.sizes()[1];
  num_rboxes = rbox_2.sizes()[0];

  at::Tensor roi_fe_2 = feat_2.new_zeros({num_rboxes, chn, out_h, out_w});
  // std::cout << feat_2.reshape(-1) << std::endl;
  roi_align_rotated_forward_cpu(feat_2, rbox_2, out_h, out_w, spatial_scale, sample_num, roi_fe_2);
  roi_fe_2 = roi_fe_2.reshape(-1);


  float diff = torch::abs(roi_fe_1 - roi_fe_2).sum().item().toFloat();
  std::cout << "roi_fe:" << diff << std::endl;

  
  // torch::Tensor outs = torch::zeros({2,256,7,7}).to(device);
  
  // int a = roi_align_rotated_forward_cpu(feat, rois, out_h, out_w, spatial_scale, sample_num, outs);

  // cout << outs[1][255] << endl;
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




