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
        self.rpn_feat = RPNFeat(in_channel, in_channel)
        # rpn head is shared with each level
        # rpn roi classification pd_scores
        self.rpn_rois_score = nn.Conv2D(
            in_channels=in_channel,
            out_channels=num_anchors,
            kernel_size=1,
            padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)))
        self.rpn_rois_score.skip_quant = True

        # rpn roi bbox regression pd_deltas
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
        pd_scores = []
        pd_deltas = []

        for rpn_feat in rpn_feats: #Region Proposal Network   Region of Interests
            score = self.rpn_rois_score(rpn_feat)#->[2, 3, 192, 336]
            delta = self.rpn_rois_delta(rpn_feat)#->[2, 12, 192, 336]
            pd_scores.append(score) #->[2, 3, 192, 336] [2, 3, 96, 168] [2, 3, 48, 84] [2, 3, 24, 42] [2, 64, 12, 21]
            pd_deltas.append(delta) #->[2, 12, 192, 336] [2, 12, 96, 168] [2, 12, 48, 84] [2, 12, 24, 42] [2, 64, 12, 21]

        anchors = self.anchor_generator(rpn_feats) #->[193536, 4] [48384, 4] [12096, 4] [3024, 4] [756, 4]

        bs_rois, bs_rois_num = self._gen_proposal(pd_scores, pd_deltas, anchors, inputs)
        if self.training:
            loss = self.get_loss(pd_scores, pd_deltas, anchors, inputs)
            return bs_rois, bs_rois_num, loss #这个损失可以看作辅助损失？
        else:
            return bs_rois, bs_rois_num, None

    def _gen_proposal(self, pd_scores, pd_deltas, anchors, inputs): #原始信息安装层来算的
        """
        pd_scores (list[Tensor]): Multi-level pd_scores prediction
        pd_deltas (list[Tensor]): Multi-level pd_deltas prediction
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

            for rpn_score, rpn_delta, anchor in zip(pd_scores, pd_deltas,
                                                    anchors):
                onnx_rpn_rois, onnx_rpn_rois_prob, onnx_rpn_rois_num, onnx_post_nms_top_n = prop_gen(
                    pd_scores=rpn_score[0:1],
                    pd_deltas=rpn_delta[0:1],
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
            # TODO(wangguanzhong): Now bs_rois_collect in export_onnx is moved outside conditional branch
            # due to problems in dy2static of paddle. Will fix it when updating paddle framework.
            # bs_rois_collect = [onnx_topk_rois]
            # bs_rois_num_collect = paddle.shape(onnx_topk_rois)[0]

        else:
            bs_rois_collect = []
            bs_rois_num_collect = []

            batch_size = im_shape.shape[0] #2

            # Generate proposals for each level and each batch.
            # Discard batch-computing to avoid sorting bbox cross different batches.
            for i in range(batch_size): #每一张图像
                rpn_rois_list = []
                rpn_rois_scores_list = []
                rpn_rois_num_list = []

                for rpn_score, rpn_delta, anchor in zip(pd_scores, pd_deltas, anchors): #对每一层进行操作
                    rpn_rois_scores, rpn_rois, rpn_rois_num, post_nms_top_n = prop_gen( #->[933, 4] [933, 1] [1] 1000
                        pd_scores=rpn_score[i:i + 1],
                        pd_deltas=rpn_delta[i:i + 1],
                        anchors=anchor,
                        im_shape=im_shape[i:i + 1]) #对每一张照片进行操作
                    rpn_rois_scores_list.append(rpn_rois_scores)
                    rpn_rois_list.append(rpn_rois)
                    rpn_rois_num_list.append(rpn_rois_num)

                if len(pd_scores) > 1:
                    rpn_rois_scores = paddle.concat(rpn_rois_scores_list).flatten()
                    rpn_rois = paddle.concat(rpn_rois_list)
                    
                    num_rois = paddle.shape(rpn_rois_scores)[0].cast('int32')
                    if num_rois > post_nms_top_n:
                        topk_prob, topk_inds = paddle.topk(rpn_rois_scores,post_nms_top_n)
                        topk_rois = paddle.gather(rpn_rois, topk_inds) #->[1000, 4]
                    else:
                        topk_rois = rpn_rois
                        topk_prob = rpn_rois_scores
                        topk_inds = paddle.zeros(shape=[post_nms_top_n], dtype="int64")
                else:
                    topk_rois = rpn_rois_list[0]
                    topk_prob = rpn_rois_scores_list[0].flatten()

                bs_rois_collect.append(topk_rois)
                bs_rois_num_collect.append(paddle.shape(topk_rois)[0:1])

                # TODO(PIR): remove this after pir bug fixed
                rpn_rois_list = None
                rpn_rois_scores_list = None
                rpn_rois_num_list = None

            bs_rois_num_collect = paddle.concat(bs_rois_num_collect)

        if self.export_onnx:
            bs_rois = [onnx_topk_rois]
            bs_rois_num = paddle.shape(onnx_topk_rois)[0]
        else:
            bs_rois = bs_rois_collect
            bs_rois_num = bs_rois_num_collect

        return bs_rois, bs_rois_num

    def get_loss(self, pd_scores, pd_deltas, anchors, inputs):
        """
        pd_scores (list[Tensor]): Multi-level pd_scores prediction
        pd_deltas (list[Tensor]): Multi-level pd_deltas prediction
        anchors (list[Tensor]): Multi-level anchors
        inputs (dict): ground truth info, including im, gt_bbox, gt_labels
        """
        anchors = [paddle.reshape(a, shape=(-1, 4)) for a in anchors]
        anchors = paddle.concat(anchors) #->[257796, 4]

        pd_scores = [
            paddle.reshape(
                paddle.transpose(
                    v, perm=[0, 2, 3, 1]),
                shape=(v.shape[0], -1, 1)) for v in pd_scores
        ] #宽高通合并
        pd_scores = paddle.concat(pd_scores, axis=1) #->[2, 257796, 1] 层级合并

        pd_deltas = [
            paddle.reshape(
                paddle.transpose(
                    v, perm=[0, 2, 3, 1]),
                shape=(v.shape[0], -1, 4)) for v in pd_deltas
        ]
        pd_deltas = paddle.concat(pd_deltas, axis=1) #->[2, 257796, 4]
        # gt_labels, gt_bboxes, gt_deltas, bs_num_samples
        gt_labels, gt_bboxes, gt_deltas, bs_num_samples = self.rpn_target_assign(inputs, anchors)

        pd_scores = paddle.reshape(x=pd_scores, shape=(-1, ))  #->[515592]
        pd_deltas = paddle.reshape(x=pd_deltas, shape=(-1, 4)) #->[515592, 4]

        gt_labels = paddle.concat(gt_labels) #[515592]
        gt_labels.stop_gradient = True

        pos_mask = gt_labels == 1
        pos_inds = paddle.nonzero(pos_mask)

        valid_masks = gt_labels >= 0
        valid_inds = paddle.nonzero(valid_masks)

        # cls loss
        if valid_inds.shape[0] == 0:
            loss_rpn_cls = paddle.zeros([1], dtype='float32')
        else:
            pd_scores = paddle.gather(pd_scores, valid_inds)
            gt_labels = paddle.gather(gt_labels, valid_inds).cast('float32')
            gt_labels.stop_gradient = True
            loss_rpn_cls = F.binary_cross_entropy_with_logits(
                logit=pd_scores, label=gt_labels, reduction="sum")

        # reg loss
        if pos_inds.shape[0] == 0:
            loss_rpn_reg = paddle.zeros([1], dtype='float32')
        else:
            pd_deltas = paddle.gather(pd_deltas, pos_inds)
            gt_deltas = paddle.concat(gt_deltas)
            gt_deltas = paddle.gather(gt_deltas, pos_inds)
            gt_deltas.stop_gradient = True

            if self.loss_rpn_bbox is None:
                loss_rpn_reg = paddle.abs(pd_deltas - gt_deltas).sum()
            else:
                loss_rpn_reg = self.loss_rpn_bbox(pd_deltas, gt_deltas).sum()

        return {
            'loss_rpn_cls': loss_rpn_cls / bs_num_samples,
            'loss_rpn_reg': loss_rpn_reg / bs_num_samples
        }
