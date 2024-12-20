# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from ppdet.core.workspace import register

__all__ = ['COTLoss']

@register
class COTLoss(nn.Layer):
    __shared__ = ['num_classes']
    def __init__(self,
                 num_classes=80, 
                 cot_scale=1,
                 cot_lambda=1):
        super(COTLoss, self).__init__()
        self.cot_scale = cot_scale
        self.cot_lambda = cot_lambda    
        self.num_classes = num_classes    
        
    def forward(self, scores, targets, cot_relation):    
        cls_name = 'loss_bbox_cls_cot'
        loss_bbox = {}

        gt_score, gt_bboxes, tgt_gt_labels = targets
        gt_score = paddle.concat(gt_score) if len(
            gt_score) > 1 else gt_score[0]
        mask = (gt_score < self.num_classes)
        valid_inds = paddle.nonzero(gt_score >= 0).flatten()
        if valid_inds.shape[0] == 0:
            loss_bbox[cls_name] = paddle.zeros([1], dtype='float32')
        else:
            gt_score = gt_score.cast('int64')
            valid_cot_targets = []
            for i in range(gt_score.shape[0]):
                train_label = gt_score[i]
                if train_label < self.num_classes:
                    valid_cot_targets.append(cot_relation[train_label])
            coco_targets = paddle.to_tensor(valid_cot_targets)
            coco_targets.stop_gradient = True
            coco_loss = - coco_targets * F.log_softmax(scores[mask][:, :-1] * self.cot_scale)
            loss_bbox[cls_name] = self.cot_lambda * paddle.mean(paddle.sum(coco_loss, axis=-1))
        return loss_bbox
