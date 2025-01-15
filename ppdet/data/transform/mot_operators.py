# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from numbers import Integral

import cv2
import copy
import numpy as np
import random
import math

from .operators import BaseOperator, register_op
from .batch_operators import Gt2TTFTarget
from ppdet.modeling.bbox_utils import bbox_iou_np_expand
from ppdet.utils.logger import setup_logger
from .op_helper import gaussian_radius
logger = setup_logger(__name__)

__all__ = [
    'RGBReverse', 'LetterBoxResize', 'MOTRandomAffine', 'Gt2JDETargetThres',
    'Gt2JDETargetMax', 'Gt2FairMOTTarget'
]


@register_op
class RGBReverse(BaseOperator):
    """RGB to BGR, or BGR to RGB, sensitive to MOTRandomAffine
    """

    def __init__(self):
        super(RGBReverse, self).__init__()

    def apply(self, batch_i, context=None):
        img = batch_i['image']
        batch_i['image'] = np.ascontiguousarray(img[:, :, ::-1])
        return batch_i


@register_op
class LetterBoxResize(BaseOperator):
    def __init__(self, target_size):
        """
        Resize image to target size, convert normalized xywh to pixel xyxy
        format ([x_center, y_center, width, height] -> [x0, y0, x1, y1]).
        Args:
            target_size (int|list): image target size.
        """
        super(LetterBoxResize, self).__init__()
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, img, height, width, color=(127.5, 127.5, 127.5)):
        # letterbox: resize a rectangular image to a padded rectangular
        shape = img.shape[:2]  # [height, width]
        ratio_h = float(height) / shape[0]
        ratio_w = float(width) / shape[1]
        ratio = min(ratio_h, ratio_w)
        new_shape = (round(shape[1] * ratio),
                     round(shape[0] * ratio))  # [width, height]
        padw = (width - new_shape[0]) / 2
        padh = (height - new_shape[1]) / 2
        top, bottom = round(padh - 0.1), round(padh + 0.1)
        left, right = round(padw - 0.1), round(padw + 0.1)

        img = cv2.resize(
            img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)  # padded rectangular
        return img, ratio, padw, padh

    def apply_bbox(self, bbox0, h, w, ratio, padw, padh):
        bboxes = bbox0.copy()
        bboxes[:, 0] = ratio * w * (bbox0[:, 0] - bbox0[:, 2] / 2) + padw
        bboxes[:, 1] = ratio * h * (bbox0[:, 1] - bbox0[:, 3] / 2) + padh
        bboxes[:, 2] = ratio * w * (bbox0[:, 0] + bbox0[:, 2] / 2) + padw
        bboxes[:, 3] = ratio * h * (bbox0[:, 1] + bbox0[:, 3] / 2) + padh
        return bboxes

    def apply(self, batch_i, context=None):
        """ Resize the image numpy.
        """
        img = batch_i['image']
        h, w = batch_i['im_shape']
        if not isinstance(img, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(img.shape) != 3:
            from PIL import UnidentifiedImageError
            raise UnidentifiedImageError(
                '{}: image is not 3-dimensional.'.format(self))

        # apply image
        height, width = self.target_size
        img, ratio, padw, padh = self.apply_image(
            img, height=height, width=width)

        batch_i['image'] = img
        new_shape = (round(h * ratio), round(w * ratio))
        batch_i['im_shape'] = np.asarray(new_shape, dtype=np.float32)
        batch_i['scale_factor'] = np.asarray([ratio, ratio], dtype=np.float32)

        # apply bbox_i
        if 'gt_bbox' in batch_i and len(batch_i['gt_bbox']) > 0:
            batch_i['gt_bbox'] = self.apply_bbox(batch_i['gt_bbox'], h, w, ratio,
                                                padw, padh)
        return batch_i


@register_op
class MOTRandomAffine(BaseOperator):
    """ 
    Affine transform to image and coords to achieve the rotate, scale and
    shift effect for training image.

    Args:
        degrees (list[2]): the rotate range to apply, transform range is [min, max]
        translate (list[2]): the translate range to apply, transform range is [min, max]
        scale (list[2]): the scale range to apply, transform range is [min, max]
        shear (list[2]): the shear range to apply, transform range is [min, max]
        borderValue (list[3]): value used in case of a constant border when appling
            the perspective transformation
        reject_outside (bool): reject warped bounding bboxes outside of image

    Returns:
        records(dict): contain the image and coords after tranformed

    """

    def __init__(self,
                 degrees=(-5, 5),
                 translate=(0.10, 0.10),
                 scale=(0.50, 1.20),
                 shear=(-2, 2),
                 borderValue=(127.5, 127.5, 127.5),
                 reject_outside=True):
        super(MOTRandomAffine, self).__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.borderValue = borderValue
        self.reject_outside = reject_outside

    def apply(self, batch_i, context=None):
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
        border = 0  # width of added border (optional)

        img = batch_i['image']
        height, width = img.shape[0], img.shape[1]

        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (self.degrees[1] - self.degrees[0]
                               ) + self.degrees[0]
        s = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        R[:2] = cv2.getRotationMatrix2D(
            angle=a, center=(width / 2, height / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (
            random.random() * 2 - 1
        ) * self.translate[0] * height + border  # x translation (pixels)
        T[1, 2] = (
            random.random() * 2 - 1
        ) * self.translate[1] * width + border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan((random.random() *
                            (self.shear[1] - self.shear[0]) + self.shear[0]) *
                           math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan((random.random() *
                            (self.shear[1] - self.shear[0]) + self.shear[0]) *
                           math.pi / 180)  # y shear (deg)

        M = S @T @R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
        imw = cv2.warpPerspective(
            img,
            M,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
            borderValue=self.borderValue)  # BGR order borderValue

        if 'gt_bbox' in batch_i and len(batch_i['gt_bbox']) > 0:
            targets = batch_i['gt_bbox']
            n = targets.shape[0]
            points = targets.copy()
            area0 = (points[:, 2] - points[:, 0]) * (
                points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate(
                (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians)))**0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate(
                (x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            if self.reject_outside:
                np.clip(xy[:, 0], 0, width, out=xy[:, 0])
                np.clip(xy[:, 2], 0, width, out=xy[:, 2])
                np.clip(xy[:, 1], 0, height, out=xy[:, 1])
                np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            if sum(i) > 0:
                batch_i['gt_bbox'] = xy[i].astype(batch_i['gt_bbox'].dtype)
                batch_i['gt_class'] = batch_i['gt_class'][i]
                if 'difficult' in batch_i:
                    batch_i['difficult'] = batch_i['difficult'][i]
                if 'gt_reid' in batch_i:
                    batch_i['gt_reid'] = batch_i['gt_reid'][i]
                if 'is_crowd' in batch_i:
                    batch_i['is_crowd'] = batch_i['is_crowd'][i]
                batch_i['image'] = imw
                return batch_i
            else:
                return batch_i


@register_op
class Gt2JDETargetThres(BaseOperator):
    __shared__ = ['num_classes']
    """
    Generate JDE targets by groud truth data when training
    Args:
        anchors (list): anchors of JDE model
        anchor_masks (list): anchor_masks of JDE model
        downsample_ratios (list): downsample ratios of JDE model
        ide_thresh (float): thresh of identity, higher is groud truth 
        fg_thresh (float): thresh of foreground, higher is foreground
        bg_thresh (float): thresh of background, lower is background
        num_classes (int): number of classes
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 ide_thresh=0.5,
                 fg_thresh=0.5,
                 bg_thresh=0.4,
                 num_classes=1):
        super(Gt2JDETargetThres, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.ide_thresh = ide_thresh
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.num_classes = num_classes

    def generate_anchor(self, nGh, nGw, anchor_hw):
        nA = len(anchor_hw)
        yy, xx = np.meshgrid(np.arange(nGh), np.arange(nGw))

        mesh = np.stack([xx.T, yy.T], axis=0)  # [2, nGh, nGw]
        mesh = np.repeat(mesh[None, :], nA, axis=0)  # [nA, 2, nGh, nGw]

        anchor_offset_mesh = anchor_hw[:, :, None][:, :, :, None]
        anchor_offset_mesh = np.repeat(anchor_offset_mesh, nGh, axis=-2)
        anchor_offset_mesh = np.repeat(anchor_offset_mesh, nGw, axis=-1)

        anchor_mesh = np.concatenate(
            [mesh, anchor_offset_mesh], axis=1)  # [nA, 4, nGh, nGw]
        return anchor_mesh

    def encode_delta(self, gt_box_list, fg_anchor_list):
        px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                        fg_anchor_list[:, 2], fg_anchor_list[:,3]
        gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], \
                        gt_box_list[:, 2], gt_box_list[:, 3]
        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = np.log(gw / pw)
        dh = np.log(gh / ph)
        return np.stack([dx, dy, dw, dh], axis=1)

    def pad_box(self, batch_i, num_max):
        assert 'gt_bbox' in batch_i
        bbox_i = batch_i['gt_bbox']
        gt_num = len(bbox_i)
        pad_bbox = np.zeros((num_max, 4), dtype=np.float32)
        if gt_num > 0:
            pad_bbox[:gt_num, :] = bbox_i[:gt_num, :]
        batch_i['gt_bbox'] = pad_bbox
        if 'gt_score' in batch_i:
            pad_score = np.zeros((num_max, ), dtype=np.float32)
            if gt_num > 0:
                pad_score[:gt_num] = batch_i['gt_score'][:gt_num, 0]
            batch_i['gt_score'] = pad_score
        if 'difficult' in batch_i:
            pad_diff = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_diff[:gt_num] = batch_i['difficult'][:gt_num, 0]
            batch_i['difficult'] = pad_diff
        if 'is_crowd' in batch_i:
            pad_crowd = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_crowd[:gt_num] = batch_i['is_crowd'][:gt_num, 0]
            batch_i['is_crowd'] = pad_crowd
        if 'gt_reid' in batch_i:
            pad_ide = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_ide[:gt_num] = batch_i['gt_reid'][:gt_num, 0]
            batch_i['gt_reid'] = pad_ide
        return batch_i

    def __call__(self, batch, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."
        h, w = batch[0]['image'].shape[1:3]

        num_max = 0
        for batch_i in batch:
            num_max = max(num_max, len(batch_i['gt_bbox']))

        for batch_i in batch:
            gt_bbox = batch_i['gt_bbox']
            gt_reid = batch_i['gt_reid']
            for i, (anchor_hw, downsample_ratio
                    ) in enumerate(zip(self.anchors, self.downsample_ratios)):
                anchor_hw = np.array(
                    anchor_hw, dtype=np.float32) / downsample_ratio
                nA = len(anchor_hw)
                nGh, nGw = int(h / downsample_ratio), int(w / downsample_ratio)
                tbox = np.zeros((nA, nGh, nGw, 4), dtype=np.float32)
                tconf = np.zeros((nA, nGh, nGw), dtype=np.float32)
                reid_i = -np.ones((nA, nGh, nGw, 1), dtype=np.float32)

                gxy, gwh = gt_bbox[:, 0:2].copy(), gt_bbox[:, 2:4].copy()
                gxy[:, 0] = gxy[:, 0] * nGw
                gxy[:, 1] = gxy[:, 1] * nGh
                gwh[:, 0] = gwh[:, 0] * nGw
                gwh[:, 1] = gwh[:, 1] * nGh
                gxy[:, 0] = np.clip(gxy[:, 0], 0, nGw - 1)
                gxy[:, 1] = np.clip(gxy[:, 1], 0, nGh - 1)
                tboxes = np.concatenate([gxy, gwh], axis=1)

                anchor_mesh = self.generate_anchor(nGh, nGw, anchor_hw)

                anchor_list = np.transpose(anchor_mesh,
                                           (0, 2, 3, 1)).reshape(-1, 4)
                iou_pdist = bbox_iou_np_expand(
                    anchor_list, tboxes, x1y1x2y2=False)

                iou_max = np.max(iou_pdist, axis=1)
                max_gt_index = np.argmax(iou_pdist, axis=1)

                iou_map = iou_max.reshape(nA, nGh, nGw)
                gt_index_map = max_gt_index.reshape(nA, nGh, nGw)

                id_index = iou_map > self.ide_thresh
                fg_index = iou_map > self.fg_thresh
                bg_index = iou_map < self.bg_thresh
                ign_index = (iou_map < self.fg_thresh) * (
                    iou_map > self.bg_thresh)
                tconf[fg_index] = 1
                tconf[bg_index] = 0
                tconf[ign_index] = -1

                gt_index = gt_index_map[fg_index]
                gt_box_list = tboxes[gt_index]
                gt_id_list = gt_reid[gt_index_map[id_index]]

                if np.sum(fg_index) > 0:
                    reid_i[id_index] = gt_id_list

                    fg_anchor_list = anchor_list.reshape(nA, nGh, nGw,
                                                         4)[fg_index]
                    delta_target = self.encode_delta(gt_box_list,
                                                     fg_anchor_list)
                    tbox[fg_index] = delta_target

                batch_i['tbox{}'.format(i)] = tbox
                batch_i['tconf{}'.format(i)] = tconf
                batch_i['tide{}'.format(i)] = reid_i
            batch_i.pop('gt_class')
            batch_i = self.pad_box(batch_i, num_max)
        return batch


@register_op
class Gt2JDETargetMax(BaseOperator):
    __shared__ = ['num_classes']
    """
    Generate JDE targets by groud truth data when evaluating
    Args:
        anchors (list): anchors of JDE model
        anchor_masks (list): anchor_masks of JDE model
        downsample_ratios (list): downsample ratios of JDE model
        max_iou_thresh (float): iou thresh for high quality anchor
        num_classes (int): number of classes
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 max_iou_thresh=0.60,
                 num_classes=1):
        super(Gt2JDETargetMax, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.max_iou_thresh = max_iou_thresh
        self.num_classes = num_classes

    def __call__(self, batch, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."
        h, w = batch[0]['image'].shape[1:3]
        for batch_i in batch:
            gt_bbox = batch_i['gt_bbox']
            gt_reid = batch_i['gt_reid']
            for i, (anchor_hw, downsample_ratio
                    ) in enumerate(zip(self.anchors, self.downsample_ratios)):
                anchor_hw = np.array(
                    anchor_hw, dtype=np.float32) / downsample_ratio
                nA = len(anchor_hw)
                nGh, nGw = int(h / downsample_ratio), int(w / downsample_ratio)
                tbox = np.zeros((nA, nGh, nGw, 4), dtype=np.float32)
                tconf = np.zeros((nA, nGh, nGw), dtype=np.float32)
                reid_i = -np.ones((nA, nGh, nGw, 1), dtype=np.float32)

                gxy, gwh = gt_bbox[:, 0:2].copy(), gt_bbox[:, 2:4].copy()
                gxy[:, 0] = gxy[:, 0] * nGw
                gxy[:, 1] = gxy[:, 1] * nGh
                gwh[:, 0] = gwh[:, 0] * nGw
                gwh[:, 1] = gwh[:, 1] * nGh
                gi = np.clip(gxy[:, 0], 0, nGw - 1).astype(int)
                gj = np.clip(gxy[:, 1], 0, nGh - 1).astype(int)

                # iou of targets-anchors (using wh only)
                box1 = gwh
                box2 = anchor_hw[:, None, :]
                inter_area = np.minimum(box1, box2).prod(2)
                iou = inter_area / (
                    box1.prod(1) + box2.prod(2) - inter_area + 1e-16)

                # Select best iou_pred and anchor
                iou_best = iou.max(0)  # best anchor [0-2] for each target
                a = np.argmax(iou, axis=0)

                # Select best unique target-anchor combinations
                iou_order = np.argsort(-iou_best)  # best to worst

                # Unique anchor selection
                u = np.stack((gi, gj, a), 0)[:, iou_order]
                _, first_unique = np.unique(u, axis=1, return_index=True)
                mask = iou_order[first_unique]
                # best anchor must share significant commonality (iou) with target
                # TODO: examine arbitrary threshold
                idx = mask[iou_best[mask] > self.max_iou_thresh]

                if len(idx) > 0:
                    a_i, gj_i, gi_i = a[idx], gj[idx], gi[idx]
                    t_box = gt_bbox[idx]
                    t_id = gt_reid[idx]
                    if len(t_box.shape) == 1:
                        t_box = t_box.reshape(1, 4)

                    gxy, gwh = t_box[:, 0:2].copy(), t_box[:, 2:4].copy()
                    gxy[:, 0] = gxy[:, 0] * nGw
                    gxy[:, 1] = gxy[:, 1] * nGh
                    gwh[:, 0] = gwh[:, 0] * nGw
                    gwh[:, 1] = gwh[:, 1] * nGh

                    # XY coordinates
                    tbox[:, :, :, 0:2][a_i, gj_i, gi_i] = gxy - gxy.astype(int)
                    # Width and height in yolo method
                    tbox[:, :, :, 2:4][a_i, gj_i, gi_i] = np.log(gwh /
                                                                 anchor_hw[a_i])
                    tconf[a_i, gj_i, gi_i] = 1
                    reid_i[a_i, gj_i, gi_i] = t_id

                batch_i['tbox{}'.format(i)] = tbox
                batch_i['tconf{}'.format(i)] = tconf
                batch_i['tide{}'.format(i)] = reid_i


class Gt2FairMOTTarget(Gt2TTFTarget):
    __shared__ = ['num_classes']
    """
    Generate FairMOT targets by ground truth data.
    Difference between Gt2FairMOTTarget and Gt2TTFTarget are:
        1. the gaussian kernal radius to generate a heatmap.
        2. the targets needed during training.
    
    Args:
        num_classes(int): the number of classes.
        down_ratio(int): the down ratio from images to heatmap, 4 by default.
        max_objs(int): the maximum number of ground truth objects in a image, 500 by default.
    """

    def __init__(self, num_classes=1, down_ratio=4, max_objs=500):
        super(Gt2TTFTarget, self).__init__()
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.max_objs = max_objs #与其说是最大检测物体数不如说是最多轨迹数

    # def __call__(self, batch, context=None):
    #     for b_id, batch_i in enumerate(batch):
    #         outmap_h = batch_i['image'].shape[1] // self.down_ratio  #152
    #         outmap_w = batch_i['image'].shape[2] // self.down_ratio  #272

    #         heatmap = np.zeros( #->(1, 152, 272)
    #             (self.num_classes, outmap_h, outmap_w), dtype='float32')
    #         bbox_size = np.zeros((self.max_objs, 4), dtype=np.float32) #(500, 4)
    #         center_offset = np.zeros((self.max_objs, 2), dtype=np.float32) #(500, 2)
    #         index = np.zeros((self.max_objs, ), dtype=np.int64) #index具体数值 (500,)
    #         index_mask = np.zeros((self.max_objs, ), dtype=np.int32) #index_mask 1 0 是pos还是neg (500,)
    #         reid = np.zeros((self.max_objs, ), dtype=np.int64) #轨迹id (500,)
    #         bbox_xyxy = np.zeros((self.max_objs, 4), dtype=np.float32)  #(500, 4)
    #         if self.num_classes > 1:
    #             # each category corresponds to a set of track ids
    #             cls_tr_ids = np.zeros(
    #                 (self.num_classes, outmap_h, outmap_w), dtype=np.int64)
    #             cls_id_map = np.full((outmap_h, outmap_w), -1, dtype=np.int64)



    #         gt_bbox = batch_i['gt_bbox']
    #         gt_class = batch_i['gt_class']
    #         gt_reid = batch_i['gt_reid']

    #         for k in range(len(gt_bbox)): #逐个框进行操作
    #             class_i = gt_class[k][0] #0
    #             bbox_i = gt_bbox[k] #(4,)
    #             outmap_bbox = bbox_i
    #             reid_i = gt_reid[k][0]
    
    #             outmap_bbox[[0, 2]] = bbox_i[[0, 2]] * outmap_w
    #             outmap_bbox[[1, 3]] = bbox_i[[1, 3]] * outmap_h #outfeat_xywh
    #             outmap_bbox_h = outmap_bbox[3] #outfeat_h
    #             outmap_bbox_w = outmap_bbox[2] #outfeat_w
                
    #             outmap_bbox_xyxy = copy.deepcopy(outmap_bbox) #目前这个是在输出特征图上的bbox
    #             outmap_bbox_xyxy[0] = outmap_bbox_xyxy[0] - outmap_bbox_xyxy[2] / 2. #输出特征图上的x1
    #             outmap_bbox_xyxy[1] = outmap_bbox_xyxy[1] - outmap_bbox_xyxy[3] / 2. #输出特征图上的y1
    #             outmap_bbox_xyxy[2] = outmap_bbox_xyxy[0] + outmap_bbox_xyxy[2] #输出特征图上的x2
    #             outmap_bbox_xyxy[3] = outmap_bbox_xyxy[1] + outmap_bbox_xyxy[3] #输出特征图上的y2



    #             if outmap_bbox_h > 0 and outmap_bbox_w > 0:
    #                 outmap_bbox_radius = gaussian_radius((math.ceil(outmap_bbox_h), math.ceil(outmap_bbox_w)), 0.7) #半径由bbox的宽高计算得出
    #                 outmap_bbox_radius = max(0, int(outmap_bbox_radius))
    #                 outmap_bbox_cxy = np.array([outmap_bbox[0], outmap_bbox[1]], dtype=np.float32)
    #                 outmap_bbox_cxy_int = outmap_bbox_cxy.astype(np.int32)
                    
    #                 self.draw_truncate_gaussian(heatmap[class_i], outmap_bbox_cxy_int, outmap_bbox_radius,
    #                                             outmap_bbox_radius) #中心点和半径
                    
    #                 bbox_size[k] = outmap_bbox_cxy[0] - outmap_bbox_xyxy[0], outmap_bbox_cxy[1] - outmap_bbox_xyxy[1], \
    #                         outmap_bbox_xyxy[2] - outmap_bbox_cxy[0], outmap_bbox_xyxy[3] - outmap_bbox_cxy[1] #左上右下

    #                 index[k] = outmap_bbox_cxy_int[1] * outmap_w + outmap_bbox_cxy_int[0] #在特征图上的哪个位置
    #                 center_offset[k] = outmap_bbox_cxy - outmap_bbox_cxy_int #实际偏差
    #                 index_mask[k] = 1
    #                 reid[k] = reid_i
    #                 bbox_xyxy[k] = outmap_bbox_xyxy
    #                 if self.num_classes > 1:
    #                     cls_id_map[outmap_bbox_cxy_int[1], outmap_bbox_cxy_int[0]] = class_i
    #                     cls_tr_ids[class_i][outmap_bbox_cxy_int[1]][outmap_bbox_cxy_int[0]] = reid_i - 1
    #                     # track id start from 0

    #         batch_i['heatmap'] = heatmap
    #         batch_i['index'] = index
    #         batch_i['offset'] = center_offset
    #         batch_i['size'] = bbox_size
    #         batch_i['index_mask'] = index_mask
    #         batch_i['reid'] = reid
    #         if self.num_classes > 1:
    #             batch_i['cls_id_map'] = cls_id_map
    #             batch_i['cls_tr_ids'] = cls_tr_ids
    #         batch_i['bbox_xyxy'] = bbox_xyxy
    #         batch_i.pop('is_crowd', None)
    #         batch_i.pop('difficult', None)
    #         batch_i.pop('gt_class', None)
    #         batch_i.pop('gt_bbox', None)
    #         batch_i.pop('gt_score', None)
    #         batch_i.pop('gt_reid', None)
    #     return batch


    def __call__(self, batch, context=None):
        for _, batch_i in enumerate(batch):
            outmap_h = batch_i['image'].shape[1] // self.down_ratio  # 152
            outmap_w = batch_i['image'].shape[2] // self.down_ratio  # 272

            # 初始化变量
            heatmap = np.zeros((self.num_classes, outmap_h, outmap_w), dtype='float32')  # (1, 152, 272)
            bbox_size = np.zeros((self.max_objs, 4), dtype=np.float32)  # (500, 4)
            center_offset = np.zeros((self.max_objs, 2), dtype=np.float32)  # (500, 2)
            index = np.zeros((self.max_objs,), dtype=np.int64)  # (500,)
            index_mask = np.zeros((self.max_objs,), dtype=np.int32)  # (500,)
            reid = np.zeros((self.max_objs,), dtype=np.int64)  # (500,)
            bbox_xyxy = np.zeros((self.max_objs, 4), dtype=np.float32)  # (500, 4)

            if self.num_classes > 1:
                cls_tr_ids = np.zeros((self.num_classes, outmap_h, outmap_w), dtype=np.int64)
                cls_id_map = np.full((outmap_h, outmap_w), -1, dtype=np.int64)

            # 获取GT数据
            gt_bbox = batch_i['gt_bbox']
            gt_class = batch_i['gt_class']
            gt_reid = batch_i['gt_reid']

            for k in range(len(gt_bbox)):  # 逐个框进行操作
                class_i = gt_class[k][0]  # 类别
                bbox_i = gt_bbox[k]  # bbox
                reid_i = gt_reid[k][0]  # 轨迹ID

                # 计算特征图上的bbox
                outmap_bbox = np.copy(bbox_i)
                outmap_bbox[[0, 2]] *= outmap_w
                outmap_bbox[[1, 3]] *= outmap_h
                
                outmap_bbox_xyxy = np.copy(outmap_bbox)
                # 转换为 (x1, y1, x2, y2) 格式
                outmap_bbox_xyxy[0] -= outmap_bbox_xyxy[2] / 2.  # x1
                outmap_bbox_xyxy[1] -= outmap_bbox_xyxy[3] / 2.  # y1
                outmap_bbox_xyxy[2] += outmap_bbox_xyxy[0]  # x2
                outmap_bbox_xyxy[3] += outmap_bbox_xyxy[1]  # y2
                
                bbox_i[0] = np.clip(bbox_i[0], 0, outmap_w - 1)
                bbox_i[1] = np.clip(bbox_i[1], 0, outmap_h - 1)
                outmap_bbox_h, outmap_bbox_w = outmap_bbox[3], outmap_bbox[2]

                outmap_bbox_xyxy2 = np.copy(outmap_bbox)
                # 转换为 (x1, y1, x2, y2) 格式
                outmap_bbox_xyxy2[0] -= outmap_bbox_xyxy2[2] / 2.  # x1
                outmap_bbox_xyxy2[1] -= outmap_bbox_xyxy2[3] / 2.  # y1
                outmap_bbox_xyxy2[2] += outmap_bbox_xyxy2[0]  # x2
                outmap_bbox_xyxy2[3] += outmap_bbox_xyxy2[1]  # y2


                if outmap_bbox_h > 0 and outmap_bbox_w > 0:
                    outmap_bbox_radius = gaussian_radius(
                        (math.ceil(outmap_bbox_h), math.ceil(outmap_bbox_w)), 0.7
                    )
                    outmap_bbox_radius = max(0, int(outmap_bbox_radius))

                    outmap_bbox_cxy = np.array([outmap_bbox[0], outmap_bbox[1]], dtype=np.float32)
                    outmap_bbox_cxy_int = outmap_bbox_cxy.astype(np.int32)

                    # 绘制截断高斯
                    self.draw_truncate_gaussian( #相当于画一个真实值的热力图
                        heatmap[class_i], outmap_bbox_cxy_int, outmap_bbox_radius, outmap_bbox_radius
                    )

                    # 更新bbox大小和偏差
                    bbox_size[k] = (outmap_bbox_cxy[0] - outmap_bbox_xyxy[0],  #左
                                    outmap_bbox_cxy[1] - outmap_bbox_xyxy[1],  #上
                                    outmap_bbox_xyxy[2] - outmap_bbox_cxy[0],  #右
                                    outmap_bbox_xyxy[3] - outmap_bbox_cxy[1])  #下

                    index[k] = outmap_bbox_cxy_int[1] * outmap_w + outmap_bbox_cxy_int[0] #bbox中心点在特征图上的索引
                    center_offset[k] = outmap_bbox_cxy - outmap_bbox_cxy_int #偏差
                    index_mask[k] = 1 #是有效的
                    reid[k] = reid_i #第几条轨迹
                    bbox_xyxy[k] = outmap_bbox_xyxy2

                    if self.num_classes > 1:
                        cls_id_map[outmap_bbox_cxy_int[1], outmap_bbox_cxy_int[0]] = class_i
                        cls_tr_ids[class_i][outmap_bbox_cxy_int[1], outmap_bbox_cxy_int[0]] = reid_i - 1

            # 将结果保存到batch中
            batch_i['heatmap'] = heatmap
            batch_i['index'] = index
            batch_i['offset'] = center_offset
            batch_i['size'] = bbox_size
            batch_i['index_mask'] = index_mask
            batch_i['reid'] = reid
            batch_i['bbox_xyxy'] = bbox_xyxy
            
            if self.num_classes > 1:
                batch_i['cls_id_map'] = cls_id_map
                batch_i['cls_tr_ids'] = cls_tr_ids



            # 清理不再需要的数据
            batch_i.pop('is_crowd', None)
            batch_i.pop('difficult', None)
            batch_i.pop('gt_class', None)
            batch_i.pop('gt_bbox', None)
            batch_i.pop('gt_score', None)
            batch_i.pop('gt_reid', None)

        return batch
