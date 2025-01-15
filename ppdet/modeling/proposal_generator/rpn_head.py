# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal

from ppdet.core.workspace import register
from .anchor_generator import AnchorGenerator
from .target_layer import RPNTargetAssign
from .proposal_generator import ProposalGenerator
from ..cls_utils import _get_class_default_kwargs


class RPNFeat(nn.Layer):
    """
    Feature extraction in RPN head

    Args:
        in_channel (int): Input channel
        out_channel (int): Output channel
    """

    def __init__(self, in_channel=1024, out_channel=1024):
        super(RPNFeat, self).__init__()
        # rpn feat is shared with each level
        self.rpn_conv = nn.Conv2D(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            padding=1,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)))
        self.rpn_conv.skip_quant = True

    def forward(self, feats):
        rpn_feats = []
        for feat in feats:
            rpn_feats.append(F.relu(self.rpn_conv(feat)))
        return rpn_feats


@register
class RPNHead(nn.Layer):
    """
    Region Proposal Network

    Args:
        anchor_generator (dict): configure of anchor generation
        rpn_target_assign (dict): configure of rpn targets assignment
        train_proposal (dict): configure of proposals generation
            at the stage of training
        test_proposal (dict): configure of proposals generation
            at the stage of prediction
        in_channel (int): channel of input feature maps which can be
            derived by from_config
    """
    __shared__ = ['export_onnx']
    __inject__ = ['loss_rpn_bbox']

    def __init__(self,
                 anchor_generator=_get_class_default_kwargs(AnchorGenerator),
                 rpn_target_assign=_get_class_default_kwargs(RPNTargetAssign),
                 train_proposal=_get_class_default_kwargs(ProposalGenerator,
                                                          12000, 2000),
                 test_proposal=_get_class_default_kwargs(ProposalGenerator),
                 in_channel=1024,
                 export_onnx=False,
                 loss_rpn_bbox=None):
        super(RPNHead, self).__init__()
        self.anchor_generator = anchor_generator
        self.rpn_target_assign = rpn_target_assign
        self.train_proposal = train_proposal
        self.test_proposal = test_proposal
        self.export_onnx = export_onnx
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGenerator(**anchor_generator)
        if isinstance(rpn_target_assign, dict):
            self.rpn_target_assign = RPNTargetAssign(**rpn_target_assign)
        if isinstance(train_proposal, dict):
            self.train_proposal = ProposalGenerator(**train_proposal)
        if isinstance(test_proposal, dict):
            self.test_proposal = ProposalGenerator(**test_proposal)
        self.loss_rpn_bbox = loss_rpn_bbox

        num_anchors = self.anchor_generator.num_anchors
        self.rpn_feat = RPNFeat(in_channel, in_channel) #论文中的小网络
        # rpn head is shared with each level
        # rpn roi classification bs_rois_scores
        self.rpn_rois_score = nn.Conv2D(
            in_channels=in_channel,
            out_channels=num_anchors,
            kernel_size=1,
            padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)))
        self.rpn_rois_score.skip_quant = True

        # rpn roi bbox regression bs_rois_deltas
        self.rpn_rois_delta = nn.Conv2D(
            in_channels=in_channel,
            out_channels=4 * num_anchors,
            kernel_size=1,
            padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)))
        self.rpn_rois_delta.skip_quant = True

    @classmethod
    def from_config(cls, cfg, input_shape):
        # FPN share same rpn head
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channel': input_shape.channels}

    def forward(self, body_feats, inputs): #Region Proposal Network   Region of Interests
        rpn_feats = self.rpn_feat(body_feats) #->[2, 256, 192, 336] [2, 256, 96, 168] [2, 256, 48, 84] [2, 256, 24, 42] [2, 256, 12, 21] 
        bs_rois_scores = []
        bs_rois_deltas = []

        for rpn_feats_pyramid in rpn_feats: #Region Proposal Network   Region of Interests
            rois_score_pyramid = self.rpn_rois_score(rpn_feats_pyramid)#->[2, 3, 192, 336] 3表示一个坐标点三个anchor 这里是rois置信度
            rois_delta_pyramid = self.rpn_rois_delta(rpn_feats_pyramid)#->[2, 12, 192, 336]
            bs_rois_scores.append(rois_score_pyramid) #->[2, 3, 192, 336] [2, 3, 96, 168] [2, 3, 48, 84] [2, 3, 24, 42] [2, 64, 12, 21]
            bs_rois_deltas.append(rois_delta_pyramid) #->[2, 12, 192, 336] [2, 12, 96, 168] [2, 12, 48, 84] [2, 12, 24, 42] [2, 64, 12, 21]

        rpn_anchors = self.anchor_generator(rpn_feats) #->[193536, 4] [48384, 4] [12096, 4] [3024, 4] [756, 4]

        bs_rois, bs_rois_num = self._gen_proposal(bs_rois_scores, bs_rois_deltas, rpn_anchors, inputs) #类别置信+nms+topk
        if self.training:
            rpn_loss = self.get_loss(bs_rois_scores, bs_rois_deltas, rpn_anchors, inputs)
            return bs_rois, bs_rois_num, rpn_loss #
        else:
            return bs_rois, bs_rois_num, None #1000个框进入后面的分类和回归

    def _gen_proposal(self, bs_rpn_scores, bs_rpn_deltas, anchors, inputs): #原始信息安装层来算的
        """
        bs_rpn_scores (list[Tensor]): Multi-level bs_rpn_scores prediction
        bs_rpn_deltas (list[Tensor]): Multi-level bs_rpn_deltas prediction
        anchors (list[Tensor]): Multi-level anchors
        inputs (dict): ground truth info
        """
        prop_gen = self.train_proposal if self.training else self.test_proposal
        im_shape = inputs['im_shape']

        # Collect multi-level proposals for each batch
        # Get 'topk' of them as final output

        if self.export_onnx:
            # bs = 1 when exporting onnx
            onnx_rpn_rois_list = []
            onnx_rpn_rois_prob_list = []
            onnx_rpn_rois_num_list = []

            for rpn_score, rpn_delta, anchor in zip(bs_rpn_scores, bs_rpn_deltas,
                                                    anchors):
                onnx_rpn_rois, onnx_rpn_rois_prob, onnx_rpn_rois_num, onnx_post_nms_top_n = prop_gen(
                    bs_rpn_scores=rpn_score[0:1],
                    bs_rois_deltas=rpn_delta[0:1],
                    anchors=anchor,
                    im_shape=im_shape[0:1])
                onnx_rpn_rois_list.append(onnx_rpn_rois)
                onnx_rpn_rois_prob_list.append(onnx_rpn_rois_prob)
                onnx_rpn_rois_num_list.append(onnx_rpn_rois_num)

            onnx_rpn_rois = paddle.concat(onnx_rpn_rois_list)
            onnx_rpn_prob = paddle.concat(onnx_rpn_rois_prob_list).flatten()

            onnx_top_n = paddle.to_tensor(onnx_post_nms_top_n).cast('int32')
            onnx_num_rois = paddle.shape(onnx_rpn_prob)[0].cast('int32')
            k = paddle.minimum(onnx_top_n, onnx_num_rois)
            onnx_topk_prob, onnx_topk_inds = paddle.topk(onnx_rpn_prob, k)
            onnx_topk_rois = paddle.gather(onnx_rpn_rois, onnx_topk_inds)
            # TODO(wangguanzhong): Now bs_rois in export_onnx is moved outside conditional branch
            # due to problems in dy2static of paddle. Will fix it when updating paddle framework.
            # bs_rois = [onnx_topk_rois]
            # bs_rois_num = paddle.shape(onnx_topk_rois)[0]

        else:
            bs_rois = []
            bs_rois_num = []

            batch_size = im_shape.shape[0] #2

            # Generate proposals for each level and each batch.
            # Discard batch-computing to avoid sorting bbox cross different batches.
            for i in range(batch_size): #每一张图像
                rois_per_img_list = []
                rois_scores_per_img_list = []
                rois_num_per_img_list = []

                for bs_rois_scores, bs_rois_deltas, anchor in zip(bs_rpn_scores, bs_rpn_deltas, anchors): #对每一层进行操作
                    rois_scores_pyramid, rois_pyramid, rois_num_pyramid, post_nms_top_n = prop_gen( #->[933, 4] [933, 1] [1] 1000
                        bs_rois_scores=bs_rois_scores[i:i + 1],
                        bs_rois_deltas=bs_rois_deltas[i:i + 1],
                        anchors=anchor,
                        im_shape=im_shape[i:i + 1]) #对每一张照片进行操作
                    
                    # 逐层合并
                    rois_scores_per_img_list.append(rois_scores_pyramid)
                    rois_per_img_list.append(rois_pyramid)
                    rois_num_per_img_list.append(rois_num_pyramid)
                #还在层但是是在一张图像
                if len(bs_rpn_scores) > 1:
                    rois_scores_per_img = paddle.concat(rois_scores_per_img_list).flatten()
                    rois_per_img = paddle.concat(rois_per_img_list)
                    
                    rois_num_per_img = paddle.shape(rois_scores_per_img)[0].cast('int32')
                    if rois_num_per_img > post_nms_top_n:
                        rois_scores_per_img, topk_inds = paddle.topk(rois_scores_per_img,post_nms_top_n)
                        rois_per_img = paddle.gather(rois_per_img, topk_inds) #->[1000, 4]
                    else:
                        topk_inds = paddle.zeros(shape=[post_nms_top_n], dtype="int64")
                else:
                    rois_per_img = rois_per_img_list[0]
                    rois_scores_per_img = rois_scores_per_img_list[0].flatten()
                    
                # 逐图像
                bs_rois.append(rois_per_img)
                bs_rois_num.append(paddle.shape(rois_per_img)[0:1])

                # TODO(PIR): remove this after pir bug fixed
                rois_per_img_list = None
                rois_scores_per_img_list = None
                rois_num_per_img_list = None

            bs_rois_num = paddle.concat(bs_rois_num)

        if self.export_onnx:
            bs_rois = [onnx_topk_rois]
            bs_rois_num = paddle.shape(onnx_topk_rois)[0]

        return bs_rois, bs_rois_num

    # def get_loss(self, bs_rois_scores, bs_rois_deltas, anchors, inputs):
    #     """
    #     bs_rois_scores (list[Tensor]): Multi-level bs_rois_scores prediction
    #     bs_rois_deltas (list[Tensor]): Multi-level bs_rois_deltas prediction
    #     anchors (list[Tensor]): Multi-level anchors
    #     inputs (dict): ground truth info, including img, gt_bbox, bs_sample_boxes_labels
    #     """
    #     anchors = [paddle.reshape(a, shape=(-1, 4)) for a in anchors]
    #     anchors = paddle.concat(anchors) #->[257796, 4]

    #     bs_rois_scores = [
    #         paddle.reshape(
    #             paddle.transpose(
    #                 v, perm=[0, 2, 3, 1]),
    #             shape=(v.shape[0], -1, 1)) for v in bs_rois_scores
    #     ] #宽高通合并
    #     bs_rois_scores = paddle.concat(bs_rois_scores, axis=1) #->[2, 257796, 1] 层级合并

    #     bs_rois_deltas = [
    #         paddle.reshape(
    #             paddle.transpose(
    #                 v, perm=[0, 2, 3, 1]),
    #             shape=(v.shape[0], -1, 4)) for v in bs_rois_deltas
    #     ]
    #     #预测情况
    #     bs_rois_deltas = paddle.concat(bs_rois_deltas, axis=1) #->[2, 257796, 4]
    #     bs_rois_scores = paddle.reshape(x=bs_rois_scores, shape=(-1, ))  #->[515592]
    #     bs_rois_deltas = paddle.reshape(x=bs_rois_deltas, shape=(-1, 4)) #->[515592, 4]
        
        
        
    #     # 
    #     bs_sample_boxes_labels, bs_tgt_rpn_boxes, bs_tgt_rpn_delta, bs_num_samples = self.rpn_target_assign(inputs, anchors)
        
        
    #     bs_sample_boxes_labels = paddle.concat(bs_sample_boxes_labels) #[515592]
    #     bs_sample_boxes_labels.stop_gradient = True


    #     valid_masks = bs_sample_boxes_labels >= 0
    #     valid_inds = paddle.nonzero(valid_masks)
    #     # cls loss
    #     if valid_inds.shape[0] == 0:
    #         loss_rpn_cls = paddle.zeros([1], dtype='float32')
    #     else:
    #         bs_rois_scores = paddle.gather(bs_rois_scores, valid_inds)
    #         bs_sample_boxes_labels = paddle.gather(bs_sample_boxes_labels, valid_inds).cast('float32')
    #         bs_sample_boxes_labels.stop_gradient = True
    #         loss_rpn_cls = F.binary_cross_entropy_with_logits(
    #             logit=bs_rois_scores, label=bs_sample_boxes_labels, reduction="sum")
            
            
            
    #     pos_mask = bs_sample_boxes_labels == 1
    #     pos_inds = paddle.nonzero(pos_mask)
    #     # reg loss
    #     if pos_inds.shape[0] == 0:
    #         loss_rpn_reg = paddle.zeros([1], dtype='float32')
    #     else:
    #         bs_rois_deltas = paddle.gather(bs_rois_deltas, pos_inds)
    #         bs_tgt_rpn_delta = paddle.concat(bs_tgt_rpn_delta)
    #         bs_tgt_rpn_delta = paddle.gather(bs_tgt_rpn_delta, pos_inds)
    #         bs_tgt_rpn_delta.stop_gradient = True

    #         if self.loss_rpn_bbox is None:
    #             loss_rpn_reg = paddle.abs(bs_rois_deltas - bs_tgt_rpn_delta).sum()
    #         else:
    #             loss_rpn_reg = self.loss_rpn_bbox(bs_rois_deltas, bs_tgt_rpn_delta).sum()

    #     return {
    #         'loss_rpn_cls': loss_rpn_cls / bs_num_samples,
    #         'loss_rpn_reg': loss_rpn_reg / bs_num_samples
    #     }



    def prepare_anchors_and_predictions(self, anchors, bs_rois_scores, bs_rois_deltas):
        """
        处理 anchors 和预测的 scores 和 deltas，使其形状一致并拼接。
        """
        anchors = [paddle.reshape(a, shape=(-1, 4)) for a in anchors]
        anchors = paddle.concat(anchors)  # ->[257796, 4]

        # 处理 bs_rois_scores
        bs_rois_scores = [
            paddle.reshape(paddle.transpose(v, perm=[0, 2, 3, 1]), shape=(v.shape[0], -1, 1)) for v in bs_rois_scores
        ]  # 宽高通合并
        bs_rois_scores = paddle.concat(bs_rois_scores, axis=1)  # ->[2, 257796, 1] 层级合并

        # 处理 bs_rois_deltas
        bs_rois_deltas = [
            paddle.reshape(paddle.transpose(v, perm=[0, 2, 3, 1]), shape=(v.shape[0], -1, 4)) for v in bs_rois_deltas
        ]
        bs_rois_deltas = paddle.concat(bs_rois_deltas, axis=1)  # ->[2, 257796, 4]

        # 进一步调整形状
        bs_rois_scores = paddle.reshape(x=bs_rois_scores, shape=(-1, ))  # ->[515592]
        bs_rois_deltas = paddle.reshape(x=bs_rois_deltas, shape=(-1, 4))  # ->[515592, 4]

        return anchors, bs_rois_scores, bs_rois_deltas

    def compute_classification_loss(self, bs_rois_scores, bs_sample_boxes_labels,valid_inds):
        """
        计算分类损失（RPN classification loss）。
        """
        if valid_inds.shape[0] == 0:
            return paddle.zeros([1], dtype='float32')
        else:
            bs_rois_scores = paddle.gather(bs_rois_scores, valid_inds)
            bs_sample_boxes_labels = paddle.gather(bs_sample_boxes_labels, valid_inds).cast('float32')
            bs_sample_boxes_labels.stop_gradient = True
            return F.binary_cross_entropy_with_logits(
                logit=bs_rois_scores, label=bs_sample_boxes_labels, reduction="sum"
            )

    def compute_regression_loss(self, bs_rois_deltas, bs_tgt_rpn_delta, pos_inds):
        """
        计算回归损失（RPN regression loss）。
        """
        if pos_inds.shape[0] == 0:
            return paddle.zeros([1], dtype='float32')
        else:
            bs_rois_deltas = paddle.gather(bs_rois_deltas, pos_inds)
            bs_tgt_rpn_delta = paddle.concat(bs_tgt_rpn_delta)
            bs_tgt_rpn_delta = paddle.gather(bs_tgt_rpn_delta, pos_inds)
            bs_tgt_rpn_delta.stop_gradient = True

            if self.loss_rpn_bbox is None:
                return paddle.abs(bs_rois_deltas - bs_tgt_rpn_delta).sum()
            else:
                return self.loss_rpn_bbox(bs_rois_deltas, bs_tgt_rpn_delta).sum()

    def get_loss(self, bs_rois_scores, bs_rois_deltas, anchors, inputs):
        """
        计算目标检测损失（包括分类损失和回归损失）。
        """
        # Step 1: 处理 anchors 和预测
        anchors, bs_rois_scores, bs_rois_deltas = self.prepare_anchors_and_predictions(anchors, bs_rois_scores, bs_rois_deltas)
        
        # Step 2: 获取目标框的标签、回归目标等
        bs_sample_boxes_labels, bs_tgt_rpn_boxes, bs_tgt_rpn_delta, bs_num_samples = self.rpn_target_assign(inputs, anchors)
        
        bs_sample_boxes_labels = paddle.concat(bs_sample_boxes_labels)  # [515592] 512有效 256正
        bs_sample_boxes_labels.stop_gradient = True

        # Step 3: 计算分类损失
        valid_masks = bs_sample_boxes_labels >= 0
        valid_inds = paddle.nonzero(valid_masks)
        loss_rpn_cls = self.compute_classification_loss(bs_rois_scores, bs_sample_boxes_labels,valid_inds)
        
        # Step 4: 计算回归损失
        pos_mask = bs_sample_boxes_labels == 1
        pos_inds = paddle.nonzero(pos_mask)
        loss_rpn_reg = self.compute_regression_loss(bs_rois_deltas, bs_tgt_rpn_delta, pos_inds)

        return {
            'loss_rpn_cls': loss_rpn_cls / bs_num_samples,
            'loss_rpn_reg': loss_rpn_reg / bs_num_samples
        }
