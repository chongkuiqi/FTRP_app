#ifndef ROI_ALIGN_ROTATED_CPU_H
#define ROI_ALIGN_ROTATED_CPU_H
#include <ATen/ATen.h>
#include <torch/torch.h>

void roi_align_rotated_forward_cpu(torch::Tensor features, torch::Tensor rois,
                           int pooled_height, int pooled_width,
                           float spatial_scale, int sample_num,
                           torch::Tensor output);


void ROIAlignRotatedForwardLaucher(const torch::Tensor features, const torch::Tensor rois,
                            const float spatial_scale, const int sample_num,
                            const int channels, const int height,
                            const int width, const int num_rois,
                            const int pooled_height, const int pooled_width,
                            torch::Tensor output);


template <typename scalar_t>
void ROIAlignRotatedForward(const int nthreads, const scalar_t *bottom_data,
                                const scalar_t *bottom_rois,
                                const scalar_t spatial_scale,
                                const int sample_num, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                scalar_t *top_data);

template <typename scalar_t>
scalar_t bilinear_interpolate(const scalar_t *bottom_data,
                                const int height, const int width,
                                scalar_t y, scalar_t x);


#endif // ROI_ALIGN_ROTATED_CPU_H
