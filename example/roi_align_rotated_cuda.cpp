// #include "roi_align_rotated_cuda.h"
// #include <torch/extension.h>
#include <torch/torch.h>
#include <cmath>
#include <vector>


int ROIAlignRotatedForwardLaucher(const torch::Tensor features, const torch::Tensor rois,
                            const float spatial_scale, const int sample_num,
                            const int channels, const int height,
                            const int width, const int num_rois,
                            const int pooled_height, const int pooled_width,
                            torch::Tensor output);

// int ROIAlignRotatedBackwardLaucher(const torch::Tensor top_grad, const torch::Tensor rois,
//                                    const float spatial_scale, const int sample_num,
//                                    const int channels, const int height,
//                                    const int width, const int num_rois,
//                                    const int pooled_height, const int pooled_width,
//                                    torch::Tensor bottom_grad);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int roi_align_rotated_forward_cuda(torch::Tensor features, torch::Tensor rois,
                           int pooled_height, int pooled_width,
                           float spatial_scale, int sample_num,
                           torch::Tensor output) {
  CHECK_INPUT(features);
  CHECK_INPUT(rois);
  CHECK_INPUT(output);

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 6) {
    printf("wrong roi size\n");
    return 0;
  }

  int num_channels = features.size(1);
  int data_height = features.size(2);
  int data_width = features.size(3);

  ROIAlignRotatedForwardLaucher(features, rois, spatial_scale, sample_num,
                         num_channels, data_height, data_width, num_rois,
                         pooled_height, pooled_width, output);

  return 1;
}

// int roi_align_rotated_backward_cuda(torch::Tensor top_grad, torch::Tensor rois,
//                             int pooled_height, int pooled_width,
//                             float spatial_scale, int sample_num,
//                             torch::Tensor bottom_grad) {
//   CHECK_INPUT(top_grad);
//   CHECK_INPUT(rois);
//   CHECK_INPUT(bottom_grad);

//   // Number of ROIs
//   int num_rois = rois.size(0);
//   int size_rois = rois.size(1);
//   if (size_rois != 6) {
//     printf("wrong roi size\n");
//     return 0;
//   }

//   int num_channels = bottom_grad.size(1);
//   int data_height = bottom_grad.size(2);
//   int data_width = bottom_grad.size(3);

//   ROIAlignRotatedBackwardLaucher(top_grad, rois, spatial_scale, sample_num,
//                           num_channels, data_height, data_width, num_rois,
//                           pooled_height, pooled_width, bottom_grad);

//   return 1;
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("forward", &roi_align_rotated_forward_cuda, "Roi_Align_Rotated forward (CUDA)");
//   // m.def("backward", &roi_align_rotated_backward_cuda, "Roi_Align_Rotated backward (CUDA)");
// }
