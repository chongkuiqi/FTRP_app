// #include <torch/extension.h>

#include <ATen/ATen.h>

template <typename scalar_t>
scalar_t bilinear_interpolate(const scalar_t *bottom_data,
                                         const int height, const int width,
                                         scalar_t y, scalar_t x);

template <typename scalar_t>
void ROIAlignRotatedForward(const int nthreads, const scalar_t *bottom_data,
                                const scalar_t *bottom_rois,
                                const scalar_t spatial_scale,
                                const int sample_num, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                scalar_t *top_data);

int ROIAlignRotatedForwardLaucher(const at::Tensor features, const at::Tensor rois,
                            const float spatial_scale, const int sample_num,
                            const int channels, const int height,
                            const int width, const int num_rois,
                            const int pooled_height, const int pooled_width,
                            at::Tensor output);


int roi_align_rotated_forward_cpu(at::Tensor features, at::Tensor rois,
                           int pooled_height, int pooled_width,
                           float spatial_scale, int sample_num,
                           at::Tensor output);