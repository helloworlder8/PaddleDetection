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

import numpy as np
import paddle
from ..bbox_utils import bbox2delta, bbox_overlaps


def rpn_anchor_target(anchors,
                      gt_boxes,
                      rpn_bs_per_im,
                      rpn_pos_thres,
                      rpn_neg_thres,
                      rpn_fg_fraction,
                      use_random=True,
                      batch_size=1,
                      ignore_thresh=-1,
                      is_crowd=None,
                      weights=[1., 1., 1., 1.],
                      assign_on_cpu=False):
    gt_labels = []
    gt_bboxes = []
    gt_deltas = []
    for i in range(batch_size): #一张一张的来
        gt_bbox = gt_boxes[i] #[4, 4]
        is_crowd_i = is_crowd[i] if is_crowd else None #[4, 1]
        
        
        # Step1: match anchor and gt_bbox
        sampled_labels, match_labels = label_box(anchors, gt_bbox, rpn_pos_thres, rpn_neg_thres, True,
            ignore_thresh, is_crowd_i, assign_on_cpu) #阈值筛选
        
        
        # Step2: sample anchor 
        pos_inds, neg_inds = subsample_labels(match_labels, rpn_bs_per_im, rpn_fg_fraction, 0, use_random) #采样点数目随机筛选
        # Fill with the ignore label (-1), then set pos_inds and neg_inds labels
        match_labels = paddle.full(match_labels.shape, -1, dtype='int32')
        if neg_inds.shape[0] > 0:
            match_labels = paddle.scatter(match_labels, neg_inds, paddle.zeros_like(neg_inds))
        if pos_inds.shape[0] > 0:
            match_labels = paddle.scatter(match_labels, pos_inds, paddle.ones_like(pos_inds))
        match_labels.stop_gradient = True
            
        # Step3: make output  
        if gt_bbox.shape[0] == 0:
            matched_boxes = paddle.zeros([sampled_labels.shape[0], 4])
            matched_delta = paddle.zeros([sampled_labels.shape[0], 4])
        else:
            matched_boxes = paddle.gather(gt_bbox, sampled_labels) #真实框坐标 预匹配
            matched_delta = bbox2delta(anchors, matched_boxes, weights)
            matched_boxes.stop_gradient = True
            matched_delta.stop_gradient = True
            
            

        gt_labels.append(match_labels) #克制的标签
        gt_bboxes.append(matched_boxes)
        gt_deltas.append(matched_delta)

    return gt_labels, gt_bboxes, gt_deltas


def label_box(anchors,
              gt_boxes,
              positive_overlap,
              negative_overlap,
              allow_low_quality,
              ignore_thresh,
              is_crowd=None,
              assign_on_cpu=False):
    if assign_on_cpu:
        device = paddle.device.get_device()
        paddle.set_device("cpu")
        iou = bbox_overlaps(gt_boxes, anchors)
        paddle.set_device(device)

    else:
        iou = bbox_overlaps(gt_boxes, anchors) # [4, 4] [257796, 4]->[4, 257796]
    n_gt = gt_boxes.shape[0] #->4
    if n_gt == 0 or is_crowd is None:
        n_gt_crowd = 0
    else:
        n_gt_crowd = paddle.nonzero(is_crowd).shape[0]
    if iou.shape[0] == 0 or n_gt_crowd == n_gt:
        # No truth, assign everything to background
        default_matches = paddle.full((iou.shape[1], ), 0, dtype='int64')
        default_match_labels = paddle.full((iou.shape[1], ), 0, dtype='int32')
        return default_matches, default_match_labels
    # if ignore_thresh > 0, remove anchor if it is closed to 
    # one of the crowded ground-truth
    if n_gt_crowd > 0: #忽略与拥挤区域的交并比计算
        N_a = anchors.shape[0]
        ones = paddle.ones([N_a])
        mask = is_crowd * ones

        if ignore_thresh > 0:
            crowd_iou = iou * mask
            valid = (paddle.sum((crowd_iou > ignore_thresh).cast('int32'),
                                axis=0) > 0).cast('float32')
            iou = iou * (1 - valid) - valid

        # ignore the iou between anchor and crowded ground-truth
        iou = iou * (1 - mask) - mask #->[4, 257796]

    matched_gt_scores, sampled_labels = paddle.topk(iou, k=1, axis=0) #->[1, 257796] [1, 257796]
    match_labels = paddle.full(sampled_labels.shape, -1, dtype='int32') #->[1, 257796]先全是-1
    # set ignored anchor with iou = -1
    neg_cond = paddle.logical_and(matched_gt_scores > -1, matched_gt_scores < negative_overlap) #->[1, 257796] bool
    match_labels = paddle.where(neg_cond, paddle.zeros_like(match_labels), match_labels) #>[1, 257796] 0 -1
    match_labels = paddle.where(matched_gt_scores >= positive_overlap, paddle.ones_like(match_labels), match_labels)#>[1, 257796] 0 -1 1 定义了正负样本 正1 负0 -1中间的
    
    if allow_low_quality: #iou最高的话不受影响
        highest_quality_foreach_gt = iou.max(axis=1, keepdim=True) #-》[4, 1]
        pred_inds_with_highest_quality = paddle.logical_and(iou > 0, iou == highest_quality_foreach_gt).cast('int32').sum(0, keepdim=True)
        match_labels = paddle.where(pred_inds_with_highest_quality > 0,paddle.ones_like(match_labels),match_labels)

    sampled_labels = sampled_labels.flatten()
    match_labels = match_labels.flatten()

    return sampled_labels, match_labels


def subsample_labels(labels,
                     num_samples,
                     fg_fraction,
                     bg_label=0,
                     use_random=True):
    #->[13, 1] 积极点位置
    pos_inds = paddle.nonzero(paddle.logical_and(labels != -1, labels != bg_label)) #->bool
    neg_inds = paddle.nonzero(labels == bg_label) #->[257107, 1] #消极点位置

    pos_num = int(num_samples * fg_fraction)
    pos_num = min(pos_inds.numel(), pos_num)
    neg_num = num_samples - pos_num
    neg_num = min(neg_inds.numel(), neg_num)
    if pos_num == 0 and neg_num == 0: #13  243
        pos_inds = paddle.zeros([0], dtype='int32')
        neg_inds = paddle.zeros([0], dtype='int32')
        return pos_inds, neg_inds

    # randomly select pos_inds and neg_inds examples

    neg_inds = neg_inds.cast('int32').flatten()
    neg_perm = paddle.randperm(neg_inds.numel(), dtype='int32') #permutation
    neg_perm = paddle.slice(neg_perm, axes=[0], starts=[0], ends=[neg_num])
    if use_random:
        neg_inds = paddle.gather(neg_inds, neg_perm)
    else:
        neg_inds = paddle.slice(neg_inds, axes=[0], starts=[0], ends=[neg_num])
    if pos_num == 0:
        pos_inds = paddle.zeros([0], dtype='int32')
        return pos_inds, neg_inds

    pos_inds = pos_inds.cast('int32').flatten()
    pos_perm = paddle.randperm(pos_inds.numel(), dtype='int32')
    pos_perm = paddle.slice(pos_perm, axes=[0], starts=[0], ends=[pos_num])
    if use_random:
        pos_inds = paddle.gather(pos_inds, pos_perm)
    else:
        pos_inds = paddle.slice(pos_inds, axes=[0], starts=[0], ends=[pos_num])

    return pos_inds, neg_inds


def generate_proposal_target(rpn_rois,
                             gt_classes,
                             gt_boxes,
                             num_samples_per_img,
                             fg_fraction,
                             pos_thres,
                             neg_thres,
                             num_classes,
                             ignore_thresh=-1.,
                             is_crowd=None,
                             use_random=True,
                             is_cascade=False,
                             cascade_iou=0.5,
                             assign_on_cpu=False,
                             add_gt_as_proposals=True):

    rois_with_gt_bbox = []
    tgt_classes = []
    tgt_bboxes = []
    tgt_gt_labels = []
    new_rois_num = []

    # In cascade rcnn, the threshold for foreground and background
    # is used from cascade_iou
    pos_thres = cascade_iou if is_cascade else pos_thres
    neg_thres = cascade_iou if is_cascade else neg_thres
    for i, rpn_roi in enumerate(rpn_rois):
        gt_bbox = gt_boxes[i]
        is_crowd_i = is_crowd[i] if is_crowd else None
        gt_class = paddle.squeeze(gt_classes[i], axis=-1)

        # Concat RoIs and gt boxes except cascade rcnn or none gt
        if add_gt_as_proposals and gt_bbox.shape[0] > 0:
            bbox = paddle.concat([rpn_roi, gt_bbox])
        else:
            bbox = rpn_roi

        # Step1: label bbox
        sampled_labels, match_labels = label_box(bbox, gt_bbox, pos_thres, neg_thres, False, ignore_thresh, is_crowd_i, assign_on_cpu) #哪一个标签
        
        
        # Step2: sample bbox  #哪一个位置 哪一个类别
        sampled_inds, sampled_classes = sample_bbox(sampled_labels, match_labels, gt_class, num_samples_per_img, fg_fraction, num_classes, use_random, is_cascade)

        # Step3: make output 
        bbox = bbox if is_cascade else paddle.gather(bbox, sampled_inds)
        sampled_labels = sampled_labels if is_cascade else paddle.gather(sampled_labels,sampled_inds) #哪一个标签
        
        if gt_bbox.shape[0] > 0:
            sampled_bbox = paddle.gather(gt_bbox, sampled_labels)
        else:
            num = bbox.shape[0]
            sampled_bbox = paddle.zeros([num, 4], dtype='float32')

        bbox.stop_gradient = True
        sampled_labels.stop_gradient = True
        gt_bbox.stop_gradient = True
        tgt_classes.append(sampled_classes)
        tgt_bboxes.append(sampled_bbox)
        rois_with_gt_bbox.append(bbox)
        tgt_gt_labels.append(sampled_labels)
        new_rois_num.append(paddle.shape(sampled_inds)[0:1])
    new_rois_num = paddle.concat(new_rois_num)
    return rois_with_gt_bbox, tgt_classes, tgt_bboxes, tgt_gt_labels, new_rois_num


def sample_bbox(sampled_labels, #[1004]
                match_labels, #[1004]
                gt_classes, #[4]
                num_samples_per_img, #512
                fg_fraction, #0.25
                num_classes, #4
                use_random=True,
                is_cascade=False):

    n_gt = gt_classes.shape[0]
    if n_gt == 0:
        # No truth, assign everything to background
        gt_classes = paddle.ones(sampled_labels.shape, dtype='int32') * num_classes
        #return sampled_labels, match_labels + num_classes
    else:
        gt_classes = paddle.gather(gt_classes, sampled_labels) #->[1004]
        gt_classes = paddle.where(match_labels == 0, paddle.ones_like(gt_classes) * num_classes,gt_classes)
        gt_classes = paddle.where(match_labels == -1, paddle.ones_like(gt_classes) * -1, gt_classes)
    if is_cascade:
        index = paddle.arange(sampled_labels.shape[0])
        return index, gt_classes
    num_samples_per_img = int(num_samples_per_img)

    pos_inds, neg_inds = subsample_labels(gt_classes, num_samples_per_img, fg_fraction,
                                        num_classes, use_random)
    if pos_inds.shape[0] == 0 and neg_inds.shape[0] == 0:
        # fake output labeled with -1 when all boxes are neither
        # foreground nor background
        sampled_inds = paddle.zeros([1], dtype='int32')
    else:
        sampled_inds = paddle.concat([pos_inds, neg_inds])
    sampled_classes = paddle.gather(gt_classes, sampled_inds)
    return sampled_inds, sampled_classes


def polygons_to_mask(polygons, height, width):
    """
    Convert the polygons to mask format

    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height (int): mask height
        width (int): mask width
    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    import pycocotools.mask as mask_util
    assert len(polygons) > 0, "COCOAPI does not support empty polygons"
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(np.bool_)


def rasterize_polygons_within_box(poly, box, resolution):
    w, h = box[2] - box[0], box[3] - box[1]
    polygons = [np.asarray(p, dtype=np.float64) for p in poly]
    for p in polygons:
        p[0::2] = p[0::2] - box[0]
        p[1::2] = p[1::2] - box[1]

    ratio_h = resolution / max(h, 0.1)
    ratio_w = resolution / max(w, 0.1)

    if ratio_h == ratio_w:
        for p in polygons:
            p *= ratio_h
    else:
        for p in polygons:
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h

    # 3. Rasterize the polygons with coco api
    mask = polygons_to_mask(polygons, resolution, resolution)
    mask = paddle.to_tensor(mask, dtype='int32')
    return mask


def generate_mask_target(gt_segms, rois, labels_int32, sampled_gt_inds,
                         num_classes, resolution):
    mask_rois = []
    mask_rois_num = []
    tgt_masks = []
    tgt_classes = []
    mask_index = []
    tgt_weights = []
    for k in range(len(rois)):
        labels_per_im = labels_int32[k]
        # select rois labeled with foreground
        pos_inds = paddle.nonzero(
            paddle.logical_and(labels_per_im != -1, labels_per_im !=
                               num_classes))
        has_fg = True
        # generate fake roi if foreground is empty
        if pos_inds.numel() == 0:
            has_fg = False
            pos_inds = paddle.ones([1, 1], dtype='int64')
        inds_per_im = sampled_gt_inds[k]
        inds_per_im = paddle.gather(inds_per_im, pos_inds)

        rois_per_im = rois[k]
        fg_rois = paddle.gather(rois_per_im, pos_inds)
        # Copy the foreground roi to cpu
        # to generate mask target with ground-truth
        boxes = fg_rois.numpy()
        gt_segms_per_im = gt_segms[k]

        new_segm = []
        inds_per_im = inds_per_im.numpy()
        if len(gt_segms_per_im) > 0:
            for i in inds_per_im:
                new_segm.append(gt_segms_per_im[i])
        fg_inds_new = pos_inds.reshape([-1]).numpy()
        results = []
        if len(gt_segms_per_im) > 0:
            for j in range(fg_inds_new.shape[0]):
                results.append(
                    rasterize_polygons_within_box(new_segm[j], boxes[j],
                                                  resolution))
        else:
            results.append(paddle.ones([resolution, resolution], dtype='int32'))

        fg_classes = paddle.gather(labels_per_im, pos_inds)
        weight = paddle.ones([fg_rois.shape[0]], dtype='float32')
        if not has_fg:
            # now all sampled classes are background
            # which will cause error in loss calculation,
            # make fake classes with weight of 0.
            fg_classes = paddle.zeros([1], dtype='int32')
            weight = weight - 1
        tgt_mask = paddle.stack(results)
        tgt_mask.stop_gradient = True
        fg_rois.stop_gradient = True

        mask_index.append(pos_inds)
        mask_rois.append(fg_rois)
        mask_rois_num.append(paddle.shape(fg_rois)[0:1])
        tgt_classes.append(fg_classes)
        tgt_masks.append(tgt_mask)
        tgt_weights.append(weight)

    mask_index = paddle.concat(mask_index)
    mask_rois_num = paddle.concat(mask_rois_num)
    tgt_classes = paddle.concat(tgt_classes, axis=0)
    tgt_masks = paddle.concat(tgt_masks, axis=0)
    tgt_weights = paddle.concat(tgt_weights, axis=0)

    return mask_rois, mask_rois_num, tgt_classes, tgt_masks, mask_index, tgt_weights


def libra_sample_pos(max_overlaps, max_classes, pos_inds, num_expected):
    if len(pos_inds) <= num_expected:
        return pos_inds
    else:
        unique_gt_inds = np.unique(max_classes[pos_inds])
        num_gts = len(unique_gt_inds)
        num_per_gt = int(round(num_expected / float(num_gts)) + 1)

        sampled_inds = []
        for i in unique_gt_inds:
            inds = np.nonzero(max_classes == i)[0]
            before_len = len(inds)
            inds = list(set(inds) & set(pos_inds))
            after_len = len(inds)
            if len(inds) > num_per_gt:
                inds = np.random.choice(inds, size=num_per_gt, replace=False)
            sampled_inds.extend(list(inds))  # combine as a new sampler
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(list(set(pos_inds) - set(sampled_inds)))
            assert len(sampled_inds) + len(extra_inds) == len(pos_inds), \
                "sum of sampled_inds({}) and extra_inds({}) length must be equal with pos_inds({})!".format(
                    len(sampled_inds), len(extra_inds), len(pos_inds))
            if len(extra_inds) > num_extra:
                extra_inds = np.random.choice(
                    extra_inds, size=num_extra, replace=False)
            sampled_inds.extend(extra_inds.tolist())
        elif len(sampled_inds) > num_expected:
            sampled_inds = np.random.choice(
                sampled_inds, size=num_expected, replace=False)
        return paddle.to_tensor(sampled_inds)


def libra_sample_via_interval(max_overlaps, full_set, num_expected, floor_thr,
                              num_bins, neg_thres):
    max_iou = max_overlaps.max()
    iou_interval = (max_iou - floor_thr) / num_bins
    per_num_expected = int(num_expected / num_bins)

    sampled_inds = []
    for i in range(num_bins):
        start_iou = floor_thr + i * iou_interval
        end_iou = floor_thr + (i + 1) * iou_interval

        tmp_set = set(
            np.where(
                np.logical_and(max_overlaps >= start_iou, max_overlaps <
                               end_iou))[0])
        tmp_inds = list(tmp_set & full_set)

        if len(tmp_inds) > per_num_expected:
            tmp_sampled_set = np.random.choice(
                tmp_inds, size=per_num_expected, replace=False)
        else:
            tmp_sampled_set = np.array(tmp_inds, dtype=np.int32)
        sampled_inds.append(tmp_sampled_set)

    sampled_inds = np.concatenate(sampled_inds)
    if len(sampled_inds) < num_expected:
        num_extra = num_expected - len(sampled_inds)
        extra_inds = np.array(list(full_set - set(sampled_inds)))
        assert len(sampled_inds) + len(extra_inds) == len(full_set), \
            "sum of sampled_inds({}) and extra_inds({}) length must be equal with full_set({})!".format(
                len(sampled_inds), len(extra_inds), len(full_set))

        if len(extra_inds) > num_extra:
            extra_inds = np.random.choice(extra_inds, num_extra, replace=False)
        sampled_inds = np.concatenate([sampled_inds, extra_inds])

    return sampled_inds


def libra_sample_neg(max_overlaps,
                     max_classes,
                     neg_inds,
                     num_expected,
                     floor_thr=-1,
                     floor_fraction=0,
                     num_bins=3,
                     neg_thres=0.5):
    if len(neg_inds) <= num_expected:
        return neg_inds
    else:
        # balance sampling for neg_inds samples
        neg_set = set(neg_inds.tolist())
        if floor_thr > 0:
            floor_set = set(
                np.where(
                    np.logical_and(max_overlaps >= 0, max_overlaps < floor_thr))
                [0])
            iou_sampling_set = set(np.where(max_overlaps >= floor_thr)[0])
        elif floor_thr == 0:
            floor_set = set(np.where(max_overlaps == 0)[0])
            iou_sampling_set = set(np.where(max_overlaps > floor_thr)[0])
        else:
            floor_set = set()
            iou_sampling_set = set(np.where(max_overlaps > floor_thr)[0])
            floor_thr = 0

        floor_neg_inds = list(floor_set & neg_set)
        iou_sampling_neg_inds = list(iou_sampling_set & neg_set)

        num_expected_iou_sampling = int(num_expected * (1 - floor_fraction))
        if len(iou_sampling_neg_inds) > num_expected_iou_sampling:
            if num_bins >= 2:
                iou_sampled_inds = libra_sample_via_interval(
                    max_overlaps,
                    set(iou_sampling_neg_inds), num_expected_iou_sampling,
                    floor_thr, num_bins, neg_thres)
            else:
                iou_sampled_inds = np.random.choice(
                    iou_sampling_neg_inds,
                    size=num_expected_iou_sampling,
                    replace=False)
        else:
            iou_sampled_inds = np.array(iou_sampling_neg_inds, dtype=np.int32)
        num_expected_floor = num_expected - len(iou_sampled_inds)
        if len(floor_neg_inds) > num_expected_floor:
            sampled_floor_inds = np.random.choice(
                floor_neg_inds, size=num_expected_floor, replace=False)
        else:
            sampled_floor_inds = np.array(floor_neg_inds, dtype=np.int32)
        sampled_inds = np.concatenate((sampled_floor_inds, iou_sampled_inds))
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(list(neg_set - set(sampled_inds)))
            if len(extra_inds) > num_extra:
                extra_inds = np.random.choice(
                    extra_inds, size=num_extra, replace=False)
            sampled_inds = np.concatenate((sampled_inds, extra_inds))
        return paddle.to_tensor(sampled_inds)


def libra_label_box(anchors, gt_boxes, gt_classes, positive_overlap,
                    negative_overlap, num_classes):
    # TODO: use paddle API to speed up
    gt_classes = gt_classes.numpy()
    gt_overlaps = np.zeros((anchors.shape[0], num_classes))
    sampled_labels = np.zeros((anchors.shape[0]), dtype=np.int32)
    if len(gt_boxes) > 0:
        proposal_to_gt_overlaps = bbox_overlaps(anchors, gt_boxes).numpy()
        overlaps_argmax = proposal_to_gt_overlaps.argmax(axis=1)
        overlaps_max = proposal_to_gt_overlaps.max(axis=1)
        # Boxes which with non-zero overlap with gt boxes
        overlapped_boxes_ind = np.where(overlaps_max > 0)[0]
        overlapped_boxes_gt_classes = gt_classes[overlaps_argmax[
            overlapped_boxes_ind]]

        for idx in range(len(overlapped_boxes_ind)):
            gt_overlaps[overlapped_boxes_ind[idx], overlapped_boxes_gt_classes[
                idx]] = overlaps_max[overlapped_boxes_ind[idx]]
            sampled_labels[overlapped_boxes_ind[idx]] = overlaps_argmax[
                overlapped_boxes_ind[idx]]

    gt_overlaps = paddle.to_tensor(gt_overlaps)
    sampled_labels = paddle.to_tensor(sampled_labels)

    matched_gt_scores = paddle.max(gt_overlaps, axis=1)
    match_labels = paddle.full(sampled_labels.shape, -1, dtype='int32')
    match_labels = paddle.where(matched_gt_scores < negative_overlap,
                                paddle.zeros_like(match_labels), match_labels)
    match_labels = paddle.where(matched_gt_scores >= positive_overlap,
                                paddle.ones_like(match_labels), match_labels)

    return sampled_labels, match_labels, matched_gt_scores


def libra_sample_bbox(sampled_labels,
                      match_labels,
                      matched_gt_scores,
                      gt_classes,
                      num_samples_per_img,
                      num_classes,
                      fg_fraction,
                      pos_thres,
                      neg_thres,
                      num_bins,
                      use_random=True,
                      is_cascade_rcnn=False):
    bbox = int(num_samples_per_img)
    fg_rois_per_im = int(np.round(fg_fraction * bbox))
    bg_rois_per_im = bbox - fg_rois_per_im

    if is_cascade_rcnn:
        pos_inds = paddle.nonzero(matched_gt_scores >= pos_thres)
        neg_inds = paddle.nonzero(matched_gt_scores < neg_thres)
    else:
        matched_vals_np = matched_gt_scores.numpy()
        match_labels_np = match_labels.numpy()

        # sample fg
        pos_inds = paddle.nonzero(matched_gt_scores >= pos_thres).flatten()
        fg_nums = int(np.minimum(fg_rois_per_im, pos_inds.shape[0]))
        if (pos_inds.shape[0] > fg_nums) and use_random:
            pos_inds = libra_sample_pos(matched_vals_np, match_labels_np,
                                       pos_inds.numpy(), fg_rois_per_im)
        pos_inds = pos_inds[:fg_nums]

        # sample bg
        neg_inds = paddle.nonzero(matched_gt_scores < neg_thres).flatten()
        bg_nums = int(np.minimum(bbox - fg_nums, neg_inds.shape[0]))
        if (neg_inds.shape[0] > bg_nums) and use_random:
            neg_inds = libra_sample_neg(
                matched_vals_np,
                match_labels_np,
                neg_inds.numpy(),
                bg_rois_per_im,
                num_bins=num_bins,
                neg_thres=neg_thres)
        neg_inds = neg_inds[:bg_nums]

        sampled_inds = paddle.concat([pos_inds, neg_inds])

        gt_classes = paddle.gather(gt_classes, sampled_labels)
        gt_classes = paddle.where(match_labels == 0,
                                  paddle.ones_like(gt_classes) * num_classes,
                                  gt_classes)
        gt_classes = paddle.where(match_labels == -1,
                                  paddle.ones_like(gt_classes) * -1, gt_classes)
        sampled_classes = paddle.gather(gt_classes, sampled_inds)

        return sampled_inds, sampled_classes


def libra_generate_proposal_target(rpn_rois,
                                   gt_classes,
                                   gt_boxes,
                                   num_samples_per_img,
                                   fg_fraction,
                                   pos_thres,
                                   neg_thres,
                                   num_classes,
                                   use_random=True,
                                   is_cascade_rcnn=False,
                                   max_overlaps=None,
                                   num_bins=3):

    rois_with_gt_bbox = []
    gt_labels = []
    gt_bboxes = []
    sampled_max_overlaps = []
    tgt_gt_labels = []
    new_rois_num = []

    for i, rpn_roi in enumerate(rpn_rois):
        max_overlap = max_overlaps[i] if is_cascade_rcnn else None
        gt_bbox = gt_boxes[i]
        gt_class = paddle.squeeze(gt_classes[i], axis=-1)
        if is_cascade_rcnn:
            rpn_roi = filter_roi(rpn_roi, max_overlap)
        bbox = paddle.concat([rpn_roi, gt_bbox])

        # Step1: label bbox
        sampled_labels, match_labels, matched_gt_scores = libra_label_box(
            bbox, gt_bbox, gt_class, pos_thres, neg_thres, num_classes)

        # Step2: sample bbox
        sampled_inds, sampled_classes = libra_sample_bbox(
            sampled_labels, match_labels, matched_gt_scores, gt_class, num_samples_per_img,
            num_classes, fg_fraction, pos_thres, neg_thres, num_bins,
            use_random, is_cascade_rcnn)

        # Step3: make output
        bbox = paddle.gather(bbox, sampled_inds)
        sampled_labels = paddle.gather(sampled_labels, sampled_inds)
        gt_bbox = paddle.gather(gt_bbox, sampled_labels)
        sampled_overlap = paddle.gather(matched_gt_scores, sampled_inds)

        bbox.stop_gradient = True
        sampled_labels.stop_gradient = True
        gt_bbox.stop_gradient = True
        sampled_overlap.stop_gradient = True

        gt_labels.append(sampled_classes)
        gt_bboxes.append(gt_bbox)
        rois_with_gt_bbox.append(bbox)
        sampled_max_overlaps.append(sampled_overlap)
        tgt_gt_labels.append(sampled_labels)
        new_rois_num.append(paddle.shape(sampled_inds)[0:1])
    new_rois_num = paddle.concat(new_rois_num)
    # rois_with_gt_bbox, gt_labels, gt_bboxes, tgt_gt_labels, new_rois_num
    return rois_with_gt_bbox, gt_labels, gt_bboxes, tgt_gt_labels, new_rois_num