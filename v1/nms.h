#ifndef NMS_H
#define NMS_H

#undef slots
#include "torch/torch.h"
#include <torch/script.h>
#define slots Q_SLOTS


struct det_results
{
    at::Tensor boxes;
    at::Tensor labels;
    at::Tensor scores;
};

det_results NMS(c10::List<at::Tensor> scores_levels, c10::List<at::Tensor> bboxes_levels,
         int max_before_nms=2000, float score_thr=0.05,
         float iou_thr=0.5, int max_per_img=2000);

#endif // NMS_H


