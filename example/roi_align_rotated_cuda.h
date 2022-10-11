#include <ATen/ATen.h>
//#include <THC/THCAtomics.cuh>

#include <torch/torch.h>

template <typename scalar_t>
__global__ void ROIAlignRotatedForward(const int nthreads, const scalar_t *bottom_data,
                                const scalar_t *bottom_rois,
                                const scalar_t spatial_scale,
                                const int sample_num, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                scalar_t *top_data);

