from copy import deepcopy
import torch
import torch.nn as nn

from models.anchors import AnchorGeneratorRotated
from models.init_weights import normal_init, bias_init_with_prob
from models.boxes import rboxes_encode, rboxes_decode

from utils.loss import SmoothL1Loss, FocalLoss

from utils.bbox_nms_rotated import multiclass_nms_rotated

from functools import partial

from models.backbone import DetectorBackbone
from models.neck import PAN

from models.alignconv import AlignConv

import math
from models.utils import assign_labels, split_to_levels, multi_apply

def multi_apply(func, *args, **kwargs):
    # 将函数func的参数kwargs固定，返回新的函数pfunc
    pfunc = partial(func, **kwargs) if kwargs else func
    # 这里的args表示feats和anchor_strides两个序列，map函数会分别遍历这两个序列，然后送入pfunc函数
    map_results = map(pfunc, *args)
    # return tuple(map(list, zip(*map_results)))
    return tuple(map(tuple, zip(*map_results)))



class RetinaNetHead(nn.Module):
    '''
    包括两部分：特征对齐模块(feature alignment module, FAM)、旋转检测模块(oriented detection module, ODM)
    input args:
        anchor_angles : 旋转anchor的角度设置，单位为弧度，由于S2ANet中角度的范围为[-0.25pi,0.75pi]，因此这个角度设置要格外注意
    '''
    def __init__(self, num_classes, in_channels=256, feat_channels=256, stacked_convs=2, 

        shared_head = True,
        
        # 使用置信度分支
        with_conf = True,

        # 是否使用对齐卷积进行分类
        with_alignconv = False,

        
        # 边界框的编解码方式，即xy坐标的编码，是否涉及到角度
        is_encode_relative = False,

        anchor_scales=[4],

        # anchor_ratios=[1.0],
        # anchor_angles = [0,],
        
        # anchor_ratios=[5.0, 8.0],
        anchor_ratios=[6.0],

        # anchor_angles = [0, 0.25*math.pi, 0.5*math.pi],
        # anchor_angles = [-0.25*math.pi, 0, 0.25*math.pi, 0.5*math.pi],
        anchor_angles = [-0.125*math.pi, 0.125*math.pi, 0.375*math.pi, 0.625*math.pi],


        featmap_strides=[8, 16, 32, 64, 128],

        score_thres_before_nms = 0.05,
        iou_thres_nms = 0.1,
        max_before_nms_per_level = 2000,
        max_per_img = 2000
    ):
        super().__init__()

        ## 输入图像的尺寸，主要用于计算损失时，对gt_boxes进行缩放，并且用于处理超出图像边界的anchors
        # (img_h, img_w)
        self.imgs_size = (1024, 1024)
        self.score_thres_before_nms = score_thres_before_nms # 进行NMS前阈值处理
        self.iou_thres_nms = iou_thres_nms                   # nms的iou阈值
        self.max_before_nms_per_level = max_before_nms_per_level     # 每个特征层级上进行NMS的检测框的个数
        self.max_per_img = max_per_img                       # 每张图像最多几个检测框

        self.num_classes = num_classes
        self.in_channels = in_channels      # 输入特征图的通道个数
        self.feat_channels = feat_channels  # head中间卷积层的输出通道数
        self.stacked_convs = stacked_convs  # head中间卷积层的个数，不包括最后输出结果的卷积层

        # 是否使用对齐卷积
        self.with_alignconv = with_alignconv   

        # 是否使用置信度预测分支
        self.with_conf = with_conf


        # 如果使用了对齐卷积，则必须使用置信度分支来挑选质量好的框
        if self.with_alignconv:
            assert self.with_conf

        # 是否使用共享head
        self.shared_head = shared_head
        # 旋转框xy坐标编解码是否涉及到角度
        self.is_encode_relative = is_encode_relative


        # FAM损失和ODM损失之间的平衡因子
        self.odm_balance = 1.0
        # FPN结构不同特征层级之间的平衡因子,这里先全部设置为1
        self.FPN_balance = (1.0, 1.0, 1.0, 1.0, 1.0) 
        # 分类损失和回归损失之间的平衡因子
        self.reg_balance = 1.0


        # anchor
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_angles = anchor_angles
        # 每个特征层级上的anchor的个数
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios) * len(self.anchor_angles)
        
        self.featmap_strides = featmap_strides
        
        # 特征层级个数
        self.num_levels = len(self.featmap_strides)

        # anchor的基础尺寸，即为特征层级的下采样倍数
        self.anchor_base_sizes = list(featmap_strides)
        
        self.anchor_generators = []
        # self.anchors = []
        for anchor_base_size in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGeneratorRotated(anchor_base_size, self.anchor_scales, self.anchor_ratios, angles=self.anchor_angles)
            )




        # S2ANet是基于RetinaNet的，不同特征层级共享head网络
        self._init_layers()
        self.init_weights()


        ## 损失函数是否创建的标志
        self.is_create_loss_func = False
        ## 损失函数定义，注意，论文中的损失，都是一个batch中所有样本的损失之和，然后除以正样本个数
        self.fl_gamma = 2.0
        self.fl_alpha = 0.5

        self.smoothL1_beta = 1.0 / 9.0

        

    def _init_layers(self):
        # self.relu = nn.ReLU(inplace=True)
        # FAM模块和ODM模块的分类分支和回归分支
        reg_ls = []
        cls_ls = []

        for i in range(self.stacked_convs):
            in_chs = self.in_channels if i == 0 else self.feat_channels
            reg_ls.append(
                nn.Sequential(
                    nn.Conv2d(in_chs, self.feat_channels, 
                        kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True),
                    nn.ReLU(inplace=True)
                )
            )
            cls_ls.append(
                nn.Sequential(
                    nn.Conv2d(in_chs, self.feat_channels, 
                        kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True),
                    nn.ReLU(inplace=True)
                )
            )

        
        self.reg_ls = nn.Sequential(*reg_ls)
        self.cls_ls = nn.Sequential(*cls_ls)

        # FAM模块用于输出的卷积层，很奇怪，FAM用的是1x1的卷积，而ODM模块用的是3x3的卷积
        self.reg_head = nn.Conv2d(self.feat_channels, 5*self.num_anchors, kernel_size=(1,1), padding=0, bias=True)
        
        # 如果使用对齐卷积分类，则只需要一个anchor；否则，需要所有的anchors
        if not self.with_alignconv:
            self.cls_head = nn.Conv2d(self.feat_channels, self.num_classes*self.num_anchors, kernel_size=(1,1), padding=0, bias=True)
        else:
            self.cls_head = nn.Conv2d(self.feat_channels, self.num_classes, kernel_size=(1,1), padding=0, bias=True)


        # 对齐卷积
        if self.with_alignconv:
            self.align_conv = AlignConv(self.feat_channels, self.feat_channels, kernel_size=3)


        # 置信度分支与回归分支并行，只需要一个head就好了
        if self.with_conf:
            # FAM模块用于输出的卷积层，很奇怪，FAM用的是1x1的卷积，而ODM模块用的是3x3的卷积
            self.conf_head = nn.Conv2d(self.feat_channels, 1*self.num_anchors, kernel_size=(1,1), padding=0, bias=True)


    def init_weights(self):
        
        bias_cls = bias_init_with_prob(0.01)

        for m in self.reg_ls.modules():
            t = type(m)
            if t is nn.Conv2d:
                normal_init(m, std=0.01)
        for m in self.cls_ls.modules():
            t = type(m)
            if t is nn.Conv2d:
                normal_init(m, std=0.01)
        normal_init(self.reg_head, std=0.01)
        normal_init(self.cls_head, std=0.01, bias=bias_cls)

        if self.with_alignconv:
            self.align_conv.init_weights()

        if self.with_conf:
            normal_init(self.conf_head, std=0.01, bias=bias_cls)


    def forward(self, feats, targets=None, imgs_size=None, post_process=False):
        # feats是个列表，存储每个层级的特征图; self.anchor_strides表示每个特征层级的下采样倍数
        # 返回的结果，是个元组，每个元组的元素是一个列表，具体形式如下：
        # ([从低特征层级到高层级的fam_cls_score, ...], [fam_bbox_pred,...], ...)
        p = multi_apply(self.forward_single, feats, self.featmap_strides)

        # return p
        # imgs_results_ls = self.get_bboxes(p)
        imgs_results_ls = self.get_bboxes_script(p)
        return imgs_results_ls

        

    
    # 经过一个特征层级的前向传播
    def forward_single(self, x, featmap_stride):
        # 查看是第几个特征层级，范围为P3-P7
        level_id = self.featmap_strides.index(featmap_stride)
        batch_size, _, feat_h, feat_w = x.shape
        # 高度和宽度，(H,W)
        featmap_size = (feat_h, feat_w)


        # 网络的直接输出，没有经过激活函数
        reg_feat = self.reg_ls(x)
        fam_bbox_pred = self.reg_head(reg_feat)
        
        ## 调整fam_bbox_pred和fam_cls_pred的shape
        # [B, num_anchors*5, H, W], to [B, H, W, num_anchors, 5]
        fam_bbox_pred = fam_bbox_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)


        # 初始的anchor
        # init_anchors shape ： [H, W, num_anchors, 5(以像素为单位)]
        init_grid_anchors = self.anchor_generators[level_id].gen_grid_anchors(
            featmap_size, self.featmap_strides[level_id])
        init_grid_anchors = init_grid_anchors.to(x.device)

        # 边界框解码
        bbox_pred_decode = self.get_bbox_decode(fam_bbox_pred.detach(), init_grid_anchors.clone())
            
        conf_pred = None
        refine_anchor = None
        index = None
        # 置信度分支
        if self.with_conf:
            conf_pred = self.conf_head(reg_feat)
            # [B, num_anchors*num_classes, H, W], to [B, H, W, num_anchors, 1]
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)

            # 根据初始的方形anchor，以及FAM的预测结果，得到修正后的旋转anchor
            # 这是一步边界框解码的过程，需要断开梯度的传递
            # 这里需要注意的是，fam_bbox_pred是没有经过激活函数的，就直接进行边界框解码了
            # 使用置信度挑选质量好的框
            # refine_anchor shape:[N, H, W, 5]
            refine_anchor, index, bbox_pred_decode = self.get_refine_anchors(conf_pred.detach(), bbox_pred_decode.clone())
        

        ## 分类分支
        if not self.with_alignconv:
            fam_cls_pred = self.cls_head(self.cls_ls(x))
            # [B, num_anchors*num_classes, H, W], to [B, H, W, num_anchors, num_classes]
            fam_cls_pred = fam_cls_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, self.num_anchors, -1)
        else:
            fam_cls_pred = self.cls_head(self.cls_ls(
                # 根据FPN的特征图、修正后的旋转anchor，获得对齐后的特征图
                self.align_conv(x, refine_anchor.clone(), featmap_stride)
            ))
            # [B, num_anchors*num_classes, H, W], to [B, H, W, num_anchors=1, num_classes]
            fam_cls_pred = fam_cls_pred.permute(0, 2, 3, 1).contiguous().reshape(batch_size, feat_h, feat_w, 1, -1)



        return fam_cls_pred, fam_bbox_pred, init_grid_anchors, refine_anchor, conf_pred, bbox_pred_decode, index
        # return fam_cls_pred, fam_bbox_pred, conf_pred


    def get_bbox_decode(self, fam_bbox_pred, init_grid_anchors):
        
        batch_size, feat_h, feat_w, _, _ = fam_bbox_pred.shape
        fam_bbox_pred = fam_bbox_pred.reshape(-1,5)

        num_anchors = self.num_anchors
        
        # init_grid_anchors shape: [H, W, num_anchors, 5] to [B, H, W, num_anchors, 5] 
        init_grid_anchors = init_grid_anchors[None, :].repeat(batch_size, 1, 1, 1, 1).reshape(-1,5)
        # 解码得到
        bbox_pred_decode = rboxes_decode(
                init_grid_anchors, fam_bbox_pred, is_encode_relative=self.is_encode_relative, wh_ratio_clip=1e-6)

        bbox_pred_decode = bbox_pred_decode.reshape(batch_size, feat_h, feat_w, num_anchors, -1)

        return bbox_pred_decode


    # 每个网格的位置出有多个anchors，选择置信度最高的anchors作为
    def get_refine_anchors(self, conf_pred, bbox_pred_decode):
        '''
            fam_cls_pred      : [B, H, W, num_anchors, num_classes]
            fam_bbox_pred     : [B, H, W, num_anchors, 5]
            init_grid_anchors : [H, W, num_anchors,5]
        '''
        num_anchors = self.num_anchors
        batch_size, feat_h, feat_w, _, _ = conf_pred.shape
        device = conf_pred.device

        # 如果FAM模块的每个特征点位置有多个anchors，那就根据分类分数，取这个位置点分数最大的anchor作为基准
        if num_anchors > 1:
            conf_pred = conf_pred.sigmoid()
            # 在类别通道找到最大值, fam_cls_pred_max_conf shape：[B,H,W,num_anchors]
            fam_cls_pred_max_conf, _ = conf_pred.max(dim=4)
            # 在num_anchors中找到置信度最大的anchors, max_conf_anchor_id shape :[B,H,W]
            _, max_conf_anchor_id = fam_cls_pred_max_conf.max(dim=3)
            max_conf_anchor_id = max_conf_anchor_id.reshape(-1)

            # 最大位置的索引，index:[B,H,W,num_anchors]
            # index = fam_cls_pred_max_conf == fam_cls_pred_max_conf_anchor[..., None]
            index = torch.zeros((batch_size*feat_h*feat_w, num_anchors), dtype=torch.bool, device=device)
            index[range(batch_size*feat_h*feat_w), max_conf_anchor_id] = True
            index = index.reshape(batch_size, feat_h, feat_w, num_anchors)

            # 在每个位置点的num_anchors个anchor中，只取出一个
            refine_anchor = bbox_pred_decode[index].reshape(batch_size, feat_h, feat_w, -1)
        
        # FAM模块每个位置点只有一个anchor，即S2ANet算法的设置
        elif num_anchors == 1:
            index = None
            refine_anchor = bbox_pred_decode.clone()
        else:
            raise NotImplementedError("num_anchors must be [1,inf)")

        return refine_anchor, index, bbox_pred_decode

 

    # 对网络预测结果进行后处理，包括边界框解码、NMS，获得输入图像尺寸上的边界框坐标
    def get_bboxes(self, p, module_name="fam"):
        '''
        module_name : 是使用ODM模块，还是使用FAM模块的预测结果，作为最终的检测结果

        '''
        assert module_name in ('fam', 'odm'), "must be FAM or ODM"

        batch_size = p[0][0].shape[0]
        num_level = len(p[0])

        # FAM模块
        # cls_pred shape : [B, H, W, num_anchors, num_classes]
        cls_pred = p[0]
        # bbox_pred_decode  shape : [B, H, W, num_anchors, 5]
        bbox_pred_decode = p[5]        

        # 使用置信度，作为边界框NMS的分数
        if self.with_conf:
            # 使用置信度分支
            conf_pred = p[4]

        if self.with_alignconv:
            assert self.num_anchors > 1
            # 筛选出单个anchors，以及对应的置信度
            index_all = p[-1]
            # p[4]为refine_anchors
            bbox_pred_decode = p[3]
            conf_choosed_pred = []
            for conf_pred_single_level, index_single_level in zip(conf_pred, index_all):
                
                _, feat_h, feat_w, _, _ = conf_pred_single_level.shape

                conf_choosed_pred.append(conf_pred_single_level[index_single_level].reshape(batch_size, feat_h, feat_w, -1))
                conf_pred = conf_choosed_pred

        # 检测框的结果
        imgs_results_ls = []
        for batch_id in range(batch_size):
            
            # 获得该张图像上的各个特征层级的预测结果
            scores_levels = []
            bbox_pred_decode_levels = []
            for level_id in range(num_level):
                if self.with_conf:
                    score_one_img_one_level = cls_pred[level_id][batch_id].detach().reshape(-1, self.num_classes).sigmoid() * \
                        conf_pred[level_id][batch_id].detach().reshape(-1, 1).sigmoid()
                else:
                    score_one_img_one_level = cls_pred[level_id][batch_id].detach().reshape(-1, self.num_classes).sigmoid()
                bbox_pred_deoce_one_img_one_level = bbox_pred_decode[level_id][batch_id].detach().reshape(-1, 5)

                scores_levels.append(score_one_img_one_level)
                bbox_pred_decode_levels.append(bbox_pred_deoce_one_img_one_level)

            # 进行一张图像的NMS
            det_bboxes, det_labels = self.get_bboxes_single_img(scores_levels, bbox_pred_decode_levels)

            imgs_results_ls.append((det_bboxes, det_labels))
        
        return tuple(imgs_results_ls)

    def get_bboxes_single_img(self, scores_levels, bbox_pred_decode_levels):
        
        # 在进行NMS之前，要先过滤掉过多的检测框
        # 注意！是对每个特征层级上的预测框个数进行限制，而不是对整个图像上的预测框进行限制
        max_before_nms_single_level = self.max_before_nms_per_level

        # 存储一张图像上所有特征层级的、经过过滤后的预测分数、框和anchor
        scores = []
        bboxes = []
        # 逐个特征层级进行处理
        for score_level, bbox_pred_decode_level in zip(scores_levels, bbox_pred_decode_levels):
            # 在NMS前，根据分类分数进行阈值处理，过滤掉过多的框
            if max_before_nms_single_level > 0 and score_level.shape[0] > max_before_nms_single_level:
                max_scores, _ = score_level.max(dim=1)
                _, topk_inds = max_scores.topk(max_before_nms_single_level)

                score_level = score_level[topk_inds, :] # shape:[N, num_classes]
                bbox_pred_decode_level = bbox_pred_decode_level[topk_inds, :]     # shape:[N, 5]
            
            scores.append(score_level)
            bboxes.append(bbox_pred_decode_level)
        
        ## 不同层级的预测结果拼接成一个tensor
        scores = torch.cat(scores, dim=0)
        bboxes = torch.cat(bboxes, dim=0)

        det_bboxes, det_labels = multiclass_nms_rotated(bboxes, scores, 
                score_thr = self.score_thres_before_nms, 
                iou_thr = self.iou_thres_nms, 
                max_per_img = self.max_per_img
            )
        
        return det_bboxes.contiguous(), det_labels
    

        # 对网络预测结果进行后处理，包括边界框解码、NMS，获得输入图像尺寸上的边界框坐标
    def get_bboxes_script(self, p, module_name="fam"):
        '''
        module_name : 是使用ODM模块，还是使用FAM模块的预测结果，作为最终的检测结果

        '''
        assert module_name in ('fam', 'odm'), "must be FAM or ODM"

        batch_size = p[0][0].shape[0]
        num_level = len(p[0])

        # FAM模块
        # cls_pred shape : [B, H, W, num_anchors, num_classes]
        cls_pred = p[0]
        # bbox_pred_decode  shape : [B, H, W, num_anchors, 5]
        bbox_pred_decode = p[5]        

        # 使用置信度，作为边界框NMS的分数
        if self.with_conf:
            # 使用置信度分支
            conf_pred = p[4]

        if self.with_alignconv:
            assert self.num_anchors > 1
            # 筛选出单个anchors，以及对应的置信度
            index_all = p[-1]
            # p[4]为refine_anchors
            bbox_pred_decode = p[3]
            conf_choosed_pred = []
            for conf_pred_single_level, index_single_level in zip(conf_pred, index_all):
                
                _, feat_h, feat_w, _, _ = conf_pred_single_level.shape

                conf_choosed_pred.append(conf_pred_single_level[index_single_level].reshape(batch_size, feat_h, feat_w, -1))
                conf_pred = conf_choosed_pred


        # 只有一张图像
        batch_id = 0            
        # 获得该张图像上的各个特征层级的预测结果
        scores_levels = []
        bbox_pred_decode_levels = []
        for level_id in range(num_level):
            if self.with_conf:
                score_one_img_one_level = cls_pred[level_id][batch_id].detach().reshape(-1, self.num_classes).sigmoid() * \
                    conf_pred[level_id][batch_id].detach().reshape(-1, 1).sigmoid()
            else:
                score_one_img_one_level = cls_pred[level_id][batch_id].detach().reshape(-1, self.num_classes).sigmoid()
            bbox_pred_deoce_one_img_one_level = bbox_pred_decode[level_id][batch_id].detach().reshape(-1, 5)

            scores_levels.append(score_one_img_one_level)
            bbox_pred_decode_levels.append(bbox_pred_deoce_one_img_one_level)
        
        # return (tuple(scores_levels), tuple(bbox_pred_decode_levels))
        return (scores_levels, bbox_pred_decode_levels)




def rotated_box_to_poly_torch_single(rrect):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    device = rrect.device
    x_ctr, y_ctr, width, height, angle = rrect[:5]
    tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
    rect = torch.tensor([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]], device=device)
    R = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                  [torch.sin(angle), torch.cos(angle)]], device=device)

    poly = torch.mm(R, rect)

    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    poly = torch.tensor([x0, y0, x1, y1, x2, y2, x3, y3], dtype=torch.float32)
    
    return poly


def rotated_box_to_poly_torch(rrects):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    polys = []
    for rrect in rrects:
        poly = rotated_box_to_poly_torch_single(rrect)
        polys.append(poly)
    
    polys = torch.stack(polys, dim=0).reshape(-1,8)
    return polys


   
class RetinaNet(nn.Module):
    def __init__(self, backbone_name="resnet50", num_classes=15):
        super().__init__()

        # 用于检测的每个特征层级的下采样次数
        # self.stride = [4, 8, 16, 32]
        self.stride = [8, 16, 32]
        # self.stride = [8, 16, 32, 64 ,128]
        self.nl = len(self.stride)  # 检测层的个数，即neck网络输出的特征层级的个数

        # self.backbone_out_out_indices = (1,2,3,4)
        # backbone输出C3、C4、C5三个特征图
        self.backbone = DetectorBackbone(backbone_name)

        # self.neck = FPN(
        #     in_channels=[512,1024,2048],
        #     num_outs=self.nl
        # )
        self.neck = PAN(
            in_channels=[512,1024,2048],
            num_outs=self.nl
        )

        # self.neck = BiFPN(
        #     in_channels=[512,1024,2048],
        #     # num_outs=self.nl
        # )
        
        self.head = RetinaNetHead(num_classes=num_classes, featmap_strides=self.stride)
        


    def forward(self, imgs, targets=None, post_process=False):

        outs = self.backbone(imgs)
        outs = self.neck(outs)
        
        
        imgs_results_ls = self.head(outs, post_process=post_process)
        return imgs_results_ls


# from roi_align_rotated.modules.roi_align_rotated import RoIAlignRotated
# from roi_align_rotated.functions.roi_align_rotated import get_roi_feat
# from roi_align_rotated import roi_align_rotated_cuda

from utils.roi_align_rotated import roi_align_rotated_cuda
class RetinaNet_extract(nn.Module):
    def __init__(self, backbone_name="resnet50", num_classes=15):
        super().__init__()

        # 用于检测的每个特征层级的下采样次数
        # self.stride = [4, 8, 16, 32]
        self.stride = [8, 16, 32]
        # self.stride = [8, 16, 32, 64 ,128]
        self.nl = len(self.stride)  # 检测层的个数，即neck网络输出的特征层级的个数

        # self.backbone_out_out_indices = (1,2,3,4)
        # backbone输出C3、C4、C5三个特征图
        self.backbone = DetectorBackbone(backbone_name)

        # self.neck = FPN(
        #     in_channels=[512,1024,2048],
        #     num_outs=self.nl
        # )
        self.neck = PAN(
            in_channels=[512,1024,2048],
            num_outs=self.nl
        )

        # self.neck = BiFPN(
        #     in_channels=[512,1024,2048],
        #     # num_outs=self.nl
        # )

        # self.roi_layers = nn.ModuleList(
        #     [RoIAlignRotated(out_size=7, sample_num=2, spatial_scale=1/s) for s in self.stride]
        # )
        

        


    def forward(self, imgs, targets=None, post_process=False):

        outs = self.backbone(imgs)
        outs = self.neck(outs)
        
        # rois = torch.tensor(
        #     [
        #         [0.0, 427.9353,616.8455, 119.1755,14.5517, -0.3343],
        #         [0.0, 60.4593, 156.7023, 186.1304, 22.0563, 1.5757]
        #     ],
        #     device=torch.device("cuda:0"),
        # )

        # # roi_feats = torch.cuda.FloatTensor(rois.size()[0], 256,
        # #                                        7, 7).fill_(0)
        
        # # roi_feats = get_roi_feat(outs[0], rois, out_h=7, out_w=7, num_channels=256, spatial_scale=1/8, sample_num=2)
        
        # features = outs[0]
        # out_h=7
        # out_w=7
        # num_channels=256
        # spatial_scale=1/8
        # sample_num=2
        # num_rois = rois.size(0)
        # output = features.new_zeros(num_rois, num_channels, out_h, out_w)
        
        # roi_align_rotated_cuda.forward(features, rois, out_h, out_w, spatial_scale,
        #                             sample_num, output)
        
        # roi_feats = self.roi_layers[0](outs[0], rois)
        # for level_id in range(self.nl):
        #     roi_feats = self.roi_layers[level_id](outs[level_id], rois)
        
        return outs

