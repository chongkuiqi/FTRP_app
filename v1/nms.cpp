// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include "box_iou_rotated_utils.h"
#include "nms.h"


template <typename scalar_t>
at::Tensor nms_rotated_cpu_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold) {
  // nms_rotated_cpu_kernel is modified from torchvision's nms_cpu_kernel,
  // however, the code in this function is much shorter because
  // we delegate the IoU computation for rotated boxes to
  // the single_box_iou_rotated function in box_iou_rotated_utils.h
  AT_ASSERTM(!dets.is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(
      dets.scalar_type() == scores.scalar_type(), "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong));
  }

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte));
  at::Tensor keep_t = at::zeros({ndets}, dets.options().dtype(at::kLong));

  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto keep = keep_t.data_ptr<int64_t>();
  auto order = order_t.data_ptr<int64_t>();

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1) {
      continue;
    }

    keep[num_to_keep++] = i;

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) {
        continue;
      }

      auto ovr = single_box_iou_rotated<scalar_t>(
          dets[i].data_ptr<scalar_t>(), dets[j].data_ptr<scalar_t>());
      if (ovr >= iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }
  return keep_t.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}

at::Tensor nms_rotated_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const float iou_threshold) {
  auto result = at::empty({0}, dets.options());
  auto dets_wl = at::cat({dets, labels.unsqueeze(1)}, 1);
  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms_rotated", [&] {
    result = nms_rotated_cpu_kernel<scalar_t>(dets_wl, scores, iou_threshold);
  });
  return result;
}



at::Tensor nms_rotated(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const float iou_threshold)
{
  assert(dets.is_cuda() == scores.is_cuda());
  if (dets.is_cuda()) AT_ERROR("Not compiled with GPU support");

  return nms_rotated_cpu(dets, scores, labels, iou_threshold);
}

det_results NMS(c10::List<at::Tensor> scores_levels, c10::List<at::Tensor> bboxes_levels,
         int max_before_nms=2000, float score_thr=0.05,
         float iou_thr=0.5, int max_per_img=2000)
{
    int num_levels = scores_levels.size();
    at::Tensor scores_level;
    at::Tensor bboxes_level;

    std::tuple<at::Tensor, at::Tensor> max_scores_tuple;
    at::Tensor max_scores_values;

    std::tuple<at::Tensor, at::Tensor> sorted_scores_tuple;
    at::Tensor rank_index;
    at::Tensor topk_index;

    std::vector<at::Tensor> scores_filtered;
    std::vector<at::Tensor> boxes_filtered;
    for (int level_id = 0; level_id < num_levels; level_id++)
    {

        scores_level = scores_levels[level_id];
        bboxes_level = bboxes_levels[level_id];
        int num_boxes = bboxes_level.sizes()[0];
        if ((max_before_nms > 0) && (num_boxes > max_before_nms))
        {
            // 类别分数通道取得最大值
            max_scores_tuple = scores_level.max(1);
            max_scores_values = std::get<0>(max_scores_tuple);

            // 从大到小排序
            sorted_scores_tuple = torch::sort(max_scores_values, 0, true);
            rank_index = std::get<1>(sorted_scores_tuple);

            // 0维度索引，0开始，max_before_nms_single_level结束
            topk_index = rank_index.slice(0, 0, max_before_nms);

            scores_filtered.push_back(
                scores_level.index_select(0, topk_index)
            );
            boxes_filtered.push_back(
                bboxes_level.index_select(0, topk_index)
            );
        }
        else
        {
            scores_filtered.push_back(scores_level);
            boxes_filtered.push_back(bboxes_level);
        }

    }

    // 不同特征层级上的预测框合并为一个tensor
    at::Tensor scores = torch::cat(scores_filtered, 0);
    at::Tensor boxes = torch::cat(boxes_filtered, 0);

    int num_classes = scores.sizes()[1];
    boxes = boxes.unsqueeze(1).repeat({1, num_classes, 1});

    at::Tensor mask = scores > score_thr;

    scores = scores.index({mask});
    boxes = boxes.index({mask});
    // 取出第二列
    at::Tensor labels = mask.nonzero().index({"...", 1});

    int num_boxes = boxes.sizes()[0];
    if (num_boxes>0)
    {
        // NMS过程
        at::Tensor keep = nms_rotated(boxes, scores, labels, iou_thr);
        scores = scores.index_select(0, keep);
        boxes = boxes.index_select(0, keep);
        labels = labels.index_select(0, keep);

        if (keep.sizes()[0]>max_per_img)
        {
            sorted_scores_tuple = torch::sort(scores, 0, true);
            rank_index = std::get<1>(sorted_scores_tuple);

            topk_index = rank_index.slice(0, 0, max_per_img);

            scores = scores.index_select(0, topk_index);
            boxes = boxes.index_select(0, topk_index);
            labels = labels.index_select(0, topk_index);
        }
    }

    det_results results;
    results.boxes = boxes;
    results.labels = labels;
    results.scores = scores;

    return results;
}

