#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from ppdet.core.workspace import register, serializable
from .. import ops


@register
@serializable
class ProposalGenerator(object):
    """
    Proposal generation module

    For more details, please refer to the document of generate_proposals 
    in ppdet/modeing/ops.py

    Args:
        pre_nms_top_n (int): Number of total bboxes to be kept per
            image before NMS. default 6000
        post_nms_top_n (int): Number of total bboxes to be kept per
            image after NMS. default 1000
        nms_thresh (float): Threshold in NMS. default 0.5
        min_size (flaot): Remove predicted boxes with either height or
             width < min_size. default 0.1
        eta (float): Apply in adaptive NMS, if adaptive `threshold > 0.5`,
             `adaptive_threshold = adaptive_threshold * eta` in each iteration.
             default 1.
        topk_after_collect (bool): whether to adopt topk after batch 
             collection. If topk_after_collect is true, box filter will not be 
             used after NMS at each image in proposal generation. default false
    """

    def __init__(self,
                 pre_nms_top_n=12000,
                 post_nms_top_n=2000,
                 nms_thresh=.5,
                 min_size=.1,
                 eta=1.,
                 topk_after_collect=False):
        super(ProposalGenerator, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.eta = eta
        self.topk_after_collect = topk_after_collect
    #  [1, 3, 192, 336] [1, 3, 192, 336] [193536, 4] [1, 2]
    def __call__(self, bs_rois_scores, bs_rois_deltas, anchors, im_shape):

        top_n = self.pre_nms_top_n if self.topk_after_collect else self.post_nms_top_n
        variances = paddle.ones_like(anchors)
        # if False:
        if hasattr(paddle.vision.ops, "generate_proposals"):
            generate_proposals = getattr(paddle.vision.ops,
                                         "generate_proposals")
        else:
            generate_proposals = ops.generate_proposals #生成候选框函数
        rpn_rois, rois_scores_per_img, rpn_rois_num = generate_proposals( #类别过滤加非极大值抑制过滤
            bs_rois_scores, #[1, 3, 192, 336]
            bs_rois_deltas, #[1, 12, 192, 336]
            im_shape, #[1, 2]
            anchors, #[193536, 4]
            variances, #[193536, 4]
            pre_nms_top_n=self.pre_nms_top_n, #2000
            post_nms_top_n=top_n,#2000
            nms_thresh=self.nms_thresh, #0.7
            min_size=self.min_size, #0
            eta=self.eta, #1
            return_rois_num=True)

        return rois_scores_per_img, rpn_rois, rpn_rois_num, self.post_nms_top_n #[933, 4] [933, 1] [1] 1000
