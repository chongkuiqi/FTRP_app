#include "nms.h"
#include "nms_rotated.h"


det_results NMS(c10::List<at::Tensor> scores_levels, c10::List<at::Tensor> bboxes_levels,
         int max_before_nms, float score_thr,
         float iou_thr, int max_per_img)
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
