#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>


using namespace std;

#define kIMAGE_SIZE 1024
#define kCHANNELS 3

bool LoadImage(std::string file_name, cv::Mat &image)
{
  image = cv::imread(file_name); // CV_8UC3
  if (image.empty() || !image.data)
    {
      return false;
    }
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  // scale image to fit
  cv::Size scale(IMG_SIZE, IMG_SIZE);
  cv::resize(image, image, scale);

  // convert [unsigned int] to [float]
  image.convertTo(image, CV_32FC3);

  return true;
}

int main(int argc, const char *argv[]) 
{

  torch::Device deviceCUDA(torch::kCUDA, 0);
  torch::Device deviceCPU(torch::kCPU);


  if (argc != 3)
  {
    std::cerr << "Usage:classifier <path-to-exported-script-module>  <path-to-lable-file> " << std::endl;
    return -1;
  }

  std::string model_path = argv[1];
  std::string img_path = argv[2];

  cv::Mat img;
  LoadImage(img_path, img);
  //read image tensor
  auto input_tensor = torch::from_blob(
    image.data, {1, IMG_SIZE, IMG_SIZE, IMG_CHN});
  input_tensor = input_tensor.permute({0, 3, 1, 2});
  input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
  input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
  input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);
  // to GPU
  input_tensor = input_tensor.to(deviceCUDA);


  // torch::Tensor img = torch::rand({1,kCHANNELS, kIMAGE_SIZE, kIMAGE_SIZE}).to(deviceCUDA);

  // std::vector<torch::jit::IValue> inputs;
  // inputs.push_back(img);


  // cout << img.itemsize() << endl;
  cout << "input img size：" << img.sizes() << endl;



  torch::jit::script::Module module;
  module = torch::jit::load(model_path, deviceCUDA);
  // module = torch::jit::load("/home/ckq/software/example/FTRP_software/model_no_align_script.pt", deviceCUDA);

  // module = torch::jit::load("/home/ckq/software/example/FTRP_software/model_test_script.pt");

  // torch::Tensor output = module.forward(inputs).toTensor().to(deviceCPU);  
  auto output = module.forward(inputs).toTuple();

  // cout << output.sh << endl;
  // int i;
  for (int i=0; i<2; i++)
  {
    auto b = output->elements()[i];
    for(int j=0; j<3; j++){
      auto c = b.toTuple()->elements()[j].toTensor();
      cout << "返回值索引:" << i << " " << "feat_level:" << j << " " << c.sizes() << endl;
    }
    
  }

}


