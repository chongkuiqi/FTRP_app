#include "roi_align_rotated_cpu.h"
// #include <torch/extension.h>
#include <torch/torch.h>
#include <cmath>
#include <vector>


// #define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x, " must be a CUDAtensor ")
// #define CHECK_CONTIGUOUS(x) \
//   TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
// #define CHECK_INPUT(x) \
//   CHECK_CUDA(x);       \
//   CHECK_CONTIGUOUS(x)

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

bool roi_align_rotated_forward_cpu(torch::Tensor features, torch::Tensor rois,
                           int pooled_height, int pooled_width,
                           float spatial_scale, int sample_num,
                           torch::Tensor output)
{
  CHECK_INPUT(features);
  CHECK_INPUT(rois);
  CHECK_INPUT(output);

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 6)
  {
    printf("wrong roi size\n");
    return false;
  }

  int num_channels = features.size(1);
  int data_height = features.size(2);
  int data_width = features.size(3);

  ROIAlignRotatedForwardLaucher(features, rois, spatial_scale, sample_num,
                         num_channels, data_height, data_width, num_rois,
                         pooled_height, pooled_width, output);

  return true;
}

int ROIAlignRotatedForwardLaucher(const torch::Tensor features, const torch::Tensor rois,
                                const float spatial_scale, const int sample_num,
                                const int channels, const int height,
                                const int width, const int num_rois,
                                const int pooled_height, const int pooled_width,
                                torch::Tensor output) {

    const int output_size = num_rois * pooled_height * pooled_width * channels;

    AT_DISPATCH_FLOATING_TYPES(
        features.scalar_type(), "ROIAlignRotatedLaucherForward", ([&] {
            const scalar_t *bottom_data = features.data_ptr<scalar_t>();
            const scalar_t *rois_data = rois.data_ptr<scalar_t>();
            scalar_t *top_data = output.data_ptr<scalar_t>();

            ROIAlignRotatedForward<scalar_t>(
                    output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                    sample_num, channels, height, width, pooled_height,
                    pooled_width, top_data);
        }));
    // THCudaCheck(cudaGetLastError());
    return 1;
}

template <typename scalar_t>
void ROIAlignRotatedForward(const int nthreads, const scalar_t *bottom_data,
                                const scalar_t *bottom_rois,
                                const scalar_t spatial_scale,
                                const int sample_num, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                scalar_t *top_data) {
    // CUDA_1D_KERNEL_LOOP(index, nthreads)

    for (int index=0; index< nthreads; index++)
    {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const scalar_t* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    scalar_t roi_center_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_center_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_width = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_height = offset_bottom_rois[4] * spatial_scale;
    // scalar_t theta = offset_bottom_rois[5] * M_PI / 180.0;
    scalar_t theta = offset_bottom_rois[5];

    // Force malformed ROIs to be 1x1
    roi_width = (roi_width>(scalar_t)1.) ? roi_width : (scalar_t)1.;
    roi_height = (roi_height>(scalar_t)1.) ? roi_height : (scalar_t)1.;
    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    const scalar_t* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sample_num > 0)
        ? sample_num
        : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    scalar_t roi_start_h = -roi_height / 2.0;
    scalar_t roi_start_w = -roi_width / 2.0;
    scalar_t cosscalar_theta = cos(theta);
    scalar_t sinscalar_theta = sin(theta);

    // We do average (integral) pooling inside a bin
    const scalar_t count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    scalar_t output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
        const scalar_t yy = roi_start_h + ph * bin_size_h +
            static_cast<scalar_t>(iy + .5f) * bin_size_h /
                static_cast<scalar_t>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const scalar_t xx = roi_start_w + pw * bin_size_w +
            static_cast<scalar_t>(ix + .5f) * bin_size_w /
                static_cast<scalar_t>(roi_bin_grid_w);

        // Rotate by theta around the center and translate
        // scalar_t x = xx * cosscalar_theta + yy * sinscalar_theta + roi_center_w;
        // scalar_t y = yy * cosscalar_theta - xx * sinscalar_theta + roi_center_h;
        scalar_t x = xx * cosscalar_theta - yy * sinscalar_theta + roi_center_w;
        scalar_t y = xx * sinscalar_theta + yy * cosscalar_theta + roi_center_h;

        scalar_t val = bilinear_interpolate<scalar_t>(
            offset_bottom_data, height, width, y, x);
        output_val += val;
        }
    }
    output_val /= count;

    top_data[index] = output_val;
    }
}


template <typename scalar_t>
scalar_t bilinear_interpolate(const scalar_t *bottom_data,
                                         const int height, const int width,
                                         scalar_t y, scalar_t x) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;
  // do bilinear interpolation
  scalar_t lt = bottom_data[y_low * width + x_low];
  scalar_t rt = bottom_data[y_low * width + x_high];
  scalar_t lb = bottom_data[y_high * width + x_low];
  scalar_t rb = bottom_data[y_high * width + x_high];
  scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  scalar_t val = (w1 * lt + w2 * rt + w3 * lb + w4 * rb);

  return val;
}

