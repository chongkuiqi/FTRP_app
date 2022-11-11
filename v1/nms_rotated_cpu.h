#ifndef NMS_ROTATED_CPU_H
#define NMS_ROTATED_CPU_H

#pragma once
//#include <torch/extension.h>
#include <torch/types.h>


template <typename scalar_t>
at::Tensor nms_rotated_cpu_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold);


at::Tensor nms_rotated_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const float iou_threshold);


at::Tensor nms_rotated(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const float iou_threshold);


struct det_results
{
    at::Tensor boxes;
    at::Tensor labels;
    at::Tensor scores;
};


det_results NMS(c10::List<at::Tensor> scores_levels, c10::List<at::Tensor> bboxes_levels,
         int max_before_nms, float score_thr,
         float iou_thr, int max_per_img);

#endif // NMS_ROTATED_CPU_H

