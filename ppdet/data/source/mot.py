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

import os
import sys
import cv2
import glob
import numpy as np
from collections import OrderedDict, defaultdict
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from .dataset import DetDataset, _make_dataset, check_file_ext
from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


@register
@serializable
class MOTDataSet(DetDataset):
    """
    Load dataset with MOT format, only support single class MOT.

    Args:
        dataset_dir (str): root directory for dataset.
        seqs_name_list (str|list): mot data image lists, muiti-source mot dataset.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        repeat (int): repeat times for dataset, use in benchmark.

    Notes:
        MOT datasets root directory following this:
            dataset/mot
            |——————image_lists
            |        |——————caltech.train  
            |        |——————caltech.val   
            |        |——————mot16.train  
            |        |——————mot17.train  
            |        ......
            |——————Caltech
            |——————MOT17
            |——————......

        All the MOT datasets have the following structure:
            Caltech
            |——————images
            |        └——————00001.jpg
            |        |—————— ...
            |        └——————0000N.jpg
            └——————labels_with_ids
                        └——————00001.txt
                        |—————— ...
                        └——————0000N.txt
            or

            MOT17
            |——————images
            |        └——————train
            |        └——————test
            └——————labels_with_ids
                        └——————train
    """

    def __init__(self,
                 dataset_dir=None,
                 seqs_name_list=[],
                 data_fields=['image'],
                 sample_num=-1,
                 repeat=1):
        super(MOTDataSet, self).__init__(
            dataset_dir=dataset_dir, #这个变量还是不要改了，牵扯太大了
            data_fields=data_fields, #需要什么样的成员属性
            sample_num=sample_num, #采样的样本数目 int
            repeat=repeat)
        self.dataset_dir = dataset_dir
        self.seqs_name_list = seqs_name_list
        if isinstance(self.seqs_name_list, str):
            self.seqs_name_list = [self.seqs_name_list]
        self.roidbs = None
        self.cname2cid = None

    def get_anno(self):
        if self.seqs_name_list == []:
            return
        # only used to get categories and metric
        # only check first data, but the label_list of all data should be same.
        first_mot_data = self.seqs_name_list[0].split('.')[0]
        anno_file = os.path.join(self.dataset_dir, first_mot_data,
                                 'label_list.txt')
        return anno_file

    def parse_dataset(self):
        self.seqs_img_paths = OrderedDict() #不同数据集以及其中的图片
        self.seqs_img_start_ind = OrderedDict()#不同数据集中第一张图片索引
        self.seqs_label_paths = OrderedDict()#不同数据集以及其中标注
        self.seqs_reid_max = OrderedDict()
        self.seqs_reid_start_ind = OrderedDict()

        seqs_img_index = 0
        for seq_name in self.seqs_name_list: #不同的数据集
            # check every data image list
            seq_name_dir = os.path.join(self.dataset_dir, 'image_lists')
            assert os.path.isdir(seq_name_dir), \
                "The {} is not a directory.".format(seq_name_dir)

            img_lists_path = os.path.join(seq_name_dir, seq_name)
            assert os.path.exists(img_lists_path), \
                "The list path {} does not exist.".format(img_lists_path)

            # record seqs_img_paths, filter out empty ones
            with open(img_lists_path, 'r') as file:
                self.seqs_img_paths[seq_name] = file.readlines()
                self.seqs_img_paths[seq_name] = [
                    os.path.join(self.dataset_dir, x.strip())
                    for x in self.seqs_img_paths[seq_name]
                ]
                self.seqs_img_paths[seq_name] = list(
                    filter(lambda x: len(x) > 0, self.seqs_img_paths[seq_name]))

                self.seqs_img_start_ind[seq_name] = seqs_img_index #，每一个数据集中的第一个图片
                seqs_img_index += len(self.seqs_img_paths[seq_name])

            # record seqs_label_paths
            self.seqs_label_paths[seq_name] = [
                x.replace('images', 'labels_with_ids').replace(
                    '.png', '.txt').replace('.jpg', '.txt')
                for x in self.seqs_img_paths[seq_name]
            ]

        for seq_name, label_paths in self.seqs_label_paths.items(): #遍历每一个标注文件夹
            seq_reid_max = -1
            for label_path_i in label_paths:
                label_i = np.loadtxt(label_path_i)
                if len(label_i) < 1:
                    continue
                if len(label_i.shape) < 2:
                    tid_max = label_i[1]
                else:
                    tid_max = np.max(label_i[:, 1])
                if tid_max > seq_reid_max:
                    seq_reid_max = tid_max
            self.seqs_reid_max[seq_name] = int(seq_reid_max + 1)

        seq_reid_sum = 0
        for i, (k, v) in enumerate(self.seqs_reid_max.items()):
            self.seqs_reid_start_ind[k] = seq_reid_sum
            seq_reid_sum += v

        self.seqs_reid_sum = defaultdict(int)
        self.seqs_reid_sum[0] = int(seq_reid_sum + 1) 
        self.num_imgs_each_seqs = [len(x) for x in self.seqs_img_paths.values()]
        self.total_imgs = sum(self.num_imgs_each_seqs)

        logger.info('MOT dataset summary: ')
        logger.info(self.seqs_reid_max)
        logger.info('Total images: {}'.format(self.total_imgs))
        logger.info('Seqs image start index: {}'.format(self.seqs_img_start_ind))
        logger.info('Total ReId: {}'.format(self.seqs_reid_sum[0]))
        logger.info('ReId start index: {}'.format(self.seqs_reid_start_ind))

        records = []
        cname2cid = mot_label()

        for img_index in range(self.total_imgs):
            for i, (k, v) in enumerate(self.seqs_img_start_ind.items()):
                if img_index >= v:
                    seq_name = list(self.seqs_label_paths.keys())[i]
                    start_index = v
            img_file = self.seqs_img_paths[seq_name][img_index - start_index]
            lbl_file = self.seqs_label_paths[seq_name][img_index - start_index]

            if not os.path.exists(img_file):
                logger.warning('Illegal image file: {}, and it will be ignored'.
                               format(img_file))
                continue
            if not os.path.isfile(lbl_file):
                logger.warning('Illegal label file: {}, and it will be ignored'.
                               format(lbl_file))
                continue

            labels = np.loadtxt(lbl_file, dtype=np.float32).reshape(-1, 6)
            # each row in labels (N, 6) is [gt_class, gt_identity, cxn, cyn, wn, hn]

            cxn, cyn = labels[:, 2], labels[:, 3]
            wn, hn = labels[:, 4], labels[:, 5]
            gt_bbox = np.stack((cxn, cyn, wn, hn)).T.astype('float32')
            gt_class = labels[:, 0:1].astype('int32')
            gt_score = np.ones((len(labels), 1)).astype('float32')
            gt_reid = labels[:, 1:2].astype('int32')
            for i, _ in enumerate(gt_reid):
                if gt_reid[i] > -1:
                    gt_reid[i] += self.seqs_reid_start_ind[seq_name]

            mot_rec = {
                'im_file': img_file, #图像文件路径
                'im_id': img_index, #图像id
            } if 'image' in self.data_fields else {}

            gt_rec = {
                'gt_class': gt_class,
                'gt_score': gt_score,
                'gt_bbox': gt_bbox,
                'gt_reid': gt_reid,
            }
            # 单类别追踪，把所有追踪对象当成一个类
            for k, v in gt_rec.items():
                if k in self.data_fields:
                    mot_rec[k] = v

            records.append(mot_rec)
            if self.sample_num > 0 and img_index >= self.sample_num:
                break
        assert len(records) > 0, 'not found any mot record in %s' % (
            self.seqs_name_list)
        self.roidbs, self.cname2cid = records, cname2cid
        # 包含的信息又 im_file im_id gt_class gt_bbox gt_reid 所有都看错一个类别person

@register
@serializable
class MCMOTDataSet(DetDataset):
    """
    Load dataset with MOT format, support multi-class MOT.

    Args:
        dataset_dir (str): root directory for dataset.
        seqs_name_list (list(str)): mcmot data image lists, muiti-source mcmot dataset.
        data_fields (list): key name of data dictionary, at least have 'image'.
        label_list (str): if use_default_label is False, will load
            mapping between category and class index.
        sample_num (int): number of samples to load, -1 means all.

    Notes:
        MCMOT datasets root directory following this:
            dataset/mot
            |——————image_lists
            |        |——————visdrone_mcmot.train  
            |        |——————visdrone_mcmot.val   
            visdrone_mcmot
            |——————images
            |        └——————train
            |        └——————val
            └——————labels_with_ids
                        └——————train
    """

    def __init__(self,
                 dataset_dir=None,
                 seqs_name_list=[],
                 data_fields=['image'],
                 label_list=None,
                 sample_num=-1):
        super(MCMOTDataSet, self).__init__(
            dataset_dir=dataset_dir,
            data_fields=data_fields,
            sample_num=sample_num)
        self.dataset_dir = dataset_dir
        self.seqs_name_list = seqs_name_list
        if isinstance(self.seqs_name_list, str):
            self.seqs_name_list = [self.seqs_name_list]
        self.label_list = label_list
        self.roidbs = None
        self.cname2cid = None

    def get_anno(self):
        if self.seqs_name_list == []:
            return
        # only used to get categories and metric
        # only check first data, but the label_list of all data should be same.
        first_mot_data = self.seqs_name_list[0].split('.')[0]
        anno_file = os.path.join(self.dataset_dir, first_mot_data,
                                 'label_list.txt')
        return anno_file

    def parse_dataset(self):
        self.seqs_img_paths = OrderedDict()
        self.seqs_img_start_ind = OrderedDict()
        self.seqs_label_paths = OrderedDict()
        self.seqs_reid_max = OrderedDict()
        self.tid_start_idx_of_cls_ids = defaultdict(dict)  # for MCMOT

        img_index = 0
        for seq_name in self.seqs_name_list:
            # check every data image list
            seq_name_dir = os.path.join(self.dataset_dir, 'seqs_name_list')
            assert os.path.isdir(seq_name_dir), \
                "The {} is not a directory.".format(seq_name_dir)

            img_lists_path = os.path.join(seq_name_dir, seq_name)
            assert os.path.exists(img_lists_path), \
                "The list path {} does not exist.".format(img_lists_path)

            # record seqs_img_paths, filter out empty ones
            with open(img_lists_path, 'r') as file:
                self.seqs_img_paths[seq_name] = file.readlines()
                self.seqs_img_paths[seq_name] = [
                    os.path.join(self.dataset_dir, x.strip())
                    for x in self.seqs_img_paths[seq_name]
                ]
                self.seqs_img_paths[seq_name] = list(
                    filter(lambda x: len(x) > 0, self.seqs_img_paths[seq_name]))

                self.seqs_img_start_ind[seq_name] = img_index
                img_index += len(self.seqs_img_paths[seq_name])

            # record seqs_label_paths
            self.seqs_label_paths[seq_name] = [
                x.replace('images', 'labels_with_ids').replace(
                    '.png', '.txt').replace('.jpg', '.txt')
                for x in self.seqs_img_paths[seq_name]
            ]

        for seq_name, label_paths in self.seqs_label_paths.items():
            # using max_ids_dict rather than seq_reid_max
            max_ids_dict = defaultdict(int)
            for label_path_i in label_paths:
                label_i = np.loadtxt(label_path_i)
                if len(label_i) < 1:
                    continue
                label_i = label_i.reshape(-1, 6)
                for item in label_i:
                    if item[1] > max_ids_dict[int(item[0])]:
                        # item[0]: cls_id
                        # item[1]: track id
                        max_ids_dict[int(item[0])] = int(item[1])
            # track id number
            self.seqs_reid_max[seq_name] = max_ids_dict

        last_idx_dict = defaultdict(int)
        for i, (k, v) in enumerate(self.seqs_reid_max.items()):  # each sub dataset
            for cls_id, id_num in v.items():  # v is a max_ids_dict
                self.tid_start_idx_of_cls_ids[k][cls_id] = last_idx_dict[cls_id]
                last_idx_dict[cls_id] += id_num

        self.seqs_reid_sum = defaultdict(int)
        for k, v in last_idx_dict.items():
            self.seqs_reid_sum[k] = int(v)  # total ids of each category

        self.num_imgs_each_seqs = [len(x) for x in self.seqs_img_paths.values()]
        self.total_imgs = sum(self.num_imgs_each_seqs)

        # cname2cid and cid2cname 
        cname2cid = {}
        if self.label_list is not None:
            # if use label_list for multi source mix dataset, 
            # please make sure label_list in the first sub_dataset at least.
            sub_dataset = self.seqs_name_list[0].split('.')[0]
            label_path = os.path.join(self.dataset_dir, sub_dataset,
                                      self.label_list)
            if not os.path.exists(label_path):
                logger.info(
                    "Note: label_list {} does not exists, use VisDrone 10 classes labels as default.".
                    format(label_path))
                cname2cid = visdrone_mcmot_label()
            else:
                with open(label_path, 'r') as fr:
                    label_id = 0
                    for line in fr.readlines():
                        cname2cid[line.strip()] = label_id
                        label_id += 1
        else:
            cname2cid = visdrone_mcmot_label()

        cid2cname = dict([(v, k) for (k, v) in cname2cid.items()])

        logger.info('MCMOT dataset summary: ')
        logger.info(self.seqs_reid_max)
        logger.info('Total images: {}'.format(self.total_imgs))
        logger.info('Image start index: {}'.format(self.seqs_img_start_ind))

        logger.info('Total trajectory of each category: ')
        seqs_reid_sum = sorted(
            self.seqs_reid_sum.items(), key=lambda x: x[0])
        total_IDs_all_cats = 0
        for (k, v) in seqs_reid_sum:
            logger.info('Category {} [{}] has {} IDs.'.format(k, cid2cname[k],
                                                              v))
            total_IDs_all_cats += v
        logger.info('Total trajectory of all categories: {}'.format(
            total_IDs_all_cats))

        logger.info('Trajectory start index of each category: ')
        for k, v in self.tid_start_idx_of_cls_ids.items():
            sorted_v = sorted(v.items(), key=lambda x: x[0])
            for (cls_id, start_idx) in sorted_v:
                logger.info('Start index of dataset {} category {:d} is {:d}'
                            .format(k, cls_id, start_idx))

        records = []
        for img_index in range(self.total_imgs):
            for i, (k, v) in enumerate(self.seqs_img_start_ind.items()):
                if img_index >= v:
                    seq_name = list(self.seqs_label_paths.keys())[i]
                    start_index = v
            img_file = self.seqs_img_paths[seq_name][img_index - start_index]
            lbl_file = self.seqs_label_paths[seq_name][img_index - start_index]

            if not os.path.exists(img_file):
                logger.warning('Illegal image file: {}, and it will be ignored'.
                               format(img_file))
                continue
            if not os.path.isfile(lbl_file):
                logger.warning('Illegal label file: {}, and it will be ignored'.
                               format(lbl_file))
                continue

            labels = np.loadtxt(lbl_file, dtype=np.float32).reshape(-1, 6)
            # each row in labels (N, 6) is [gt_class, gt_identity, cxn, cyn, wn, hn]

            cxn, cyn = labels[:, 2], labels[:, 3]
            wn, hn = labels[:, 4], labels[:, 5]
            gt_bbox = np.stack((cxn, cyn, wn, hn)).T.astype('float32')
            gt_class = labels[:, 0:1].astype('int32')
            gt_score = np.ones((len(labels), 1)).astype('float32')
            gt_reid = labels[:, 1:2].astype('int32')
            for i, _ in enumerate(gt_reid):
                if gt_reid[i] > -1:
                    cls_id = int(gt_class[i])
                    start_idx = self.tid_start_idx_of_cls_ids[seq_name][cls_id]
                    gt_reid[i] += start_idx

            mot_rec = {
                'im_file': img_file,
                'im_id': img_index,
            } if 'image' in self.data_fields else {}

            gt_rec = {
                'gt_class': gt_class,
                'gt_score': gt_score,
                'gt_bbox': gt_bbox,
                'gt_reid': gt_reid,
            }

            for k, v in gt_rec.items():
                if k in self.data_fields:
                    mot_rec[k] = v

            records.append(mot_rec)
            if self.sample_num > 0 and img_index >= self.sample_num:
                break
        assert len(records) > 0, 'not found any mot record in %s' % (
            self.seqs_name_list)
        self.roidbs, self.cname2cid = records, cname2cid


@register
@serializable
class MOTImageFolder(DetDataset):
    """
    Load MOT dataset with MOT format from image folder or video .
    Args:
        video_file (str): path of the video file, default ''.
        frame_rate (int): frame rate of the video, use cv2 VideoCapture if not set.
        dataset_dir (str): root directory for dataset.
        keep_ori_im (bool): whether to keep original image, default False. 
            Set True when used during MOT model inference while saving
            images or video, or used in DeepSORT.
    """

    def __init__(self,
                 video_file=None,
                 frame_rate=-1,
                 dataset_dir=None,
                 data_root=None,
                 image_dir=None,
                 sample_num=-1,
                 keep_ori_im=False,
                 anno_path=None,
                 **kwargs):
        super(MOTImageFolder, self).__init__(
            dataset_dir, image_dir, sample_num=sample_num)
        self.video_file = video_file
        self.data_root = data_root
        self.keep_ori_im = keep_ori_im
        self._imid2path = {}
        self.roidbs = None
        self.frame_rate = frame_rate
        self.anno_path = anno_path

    def check_or_download_dataset(self):
        return

    def parse_dataset(self, ):
        if not self.roidbs:
            if self.video_file is None:
                self.frame_rate = 30  # set as default if infer image folder
                self.roidbs = self._load_images()
            else:
                self.roidbs = self._load_video_images()

    def _load_video_images(self):
        if self.frame_rate == -1:
            # if frame_rate is not set for video, use cv2.VideoCapture
            cap = cv2.VideoCapture(self.video_file)
            self.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        extension = self.video_file.split('.')[-1]
        output_path = self.video_file.replace('.{}'.format(extension), '')
        frames_path = video2frames(self.video_file, output_path,
                                   self.frame_rate)
        self.video_frames = sorted(
            glob.glob(os.path.join(frames_path, '*.png')))

        self.video_length = len(self.video_frames)
        logger.info('Length of the video: {:d} frames.'.format(
            self.video_length))
        ct = 0
        records = []
        for image in self.video_frames:
            assert image != '' and os.path.isfile(image), \
                    "Image {} not found".format(image)
            if self.sample_num > 0 and ct >= self.sample_num:
                break
            rec = {'im_id': np.array([ct]), 'im_file': image}
            if self.keep_ori_im:
                rec.update({'keep_ori_im': 1})
            self._imid2path[ct] = image
            ct += 1
            records.append(rec)
        assert len(records) > 0, "No image file found"
        return records

    def _find_images(self):
        image_dir = self.image_dir
        if not isinstance(image_dir, Sequence):
            image_dir = [image_dir]
        images = []
        for im_dir in image_dir:
            if os.path.isdir(im_dir):
                im_dir = os.path.join(self.dataset_dir, im_dir)
                images.extend(_make_dataset(im_dir))
            elif os.path.isfile(im_dir) and check_file_ext(im_dir):
                images.append(im_dir)
        return images

    def _load_images(self):
        images = self._find_images()
        ct = 0
        records = []
        for image in images:
            assert image != '' and os.path.isfile(image), \
                    "Image {} not found".format(image)
            if self.sample_num > 0 and ct >= self.sample_num:
                break
            rec = {'im_id': np.array([ct]), 'im_file': image}
            if self.keep_ori_im:
                rec.update({'keep_ori_im': 1})
            self._imid2path[ct] = image
            ct += 1
            records.append(rec)
        assert len(records) > 0, "No image file found"
        return records

    def get_imid2path(self):
        return self._imid2path

    def set_images(self, images):
        self.image_dir = images
        self.roidbs = self._load_images()

    def set_video(self, video_file, frame_rate):
        # update video_file and frame_rate by command line of tools/infer_mot.py
        self.video_file = video_file
        self.frame_rate = frame_rate
        assert os.path.isfile(self.video_file) and _is_valid_video(self.video_file), \
                "wrong or unsupported file format: {}".format(self.video_file)
        self.roidbs = self._load_video_images()

    def get_anno(self):
        return self.anno_path


def _is_valid_video(f, extensions=('.mp4', '.avi', '.mov', '.rmvb', 'flv')):
    return f.lower().endswith(extensions)


def video2frames(video_path, outpath, frame_rate, **kargs):
    def _dict2str(kargs):
        cmd_str = ''
        for k, v in kargs.items():
            cmd_str += (' ' + str(k) + ' ' + str(v))
        return cmd_str

    ffmpeg = ['ffmpeg ', ' -y -loglevel ', ' error ']
    vid_name = os.path.basename(video_path).split('.')[0]
    out_full_path = os.path.join(outpath, vid_name)

    if not os.path.exists(out_full_path):
        os.makedirs(out_full_path)

    # video file name
    outformat = os.path.join(out_full_path, '%08d.png')

    cmd = ffmpeg
    cmd = ffmpeg + [
        ' -i ', video_path, ' -r ', str(frame_rate), ' -f image2 ', outformat
    ]
    cmd = ''.join(cmd) + _dict2str(kargs)

    if os.system(cmd) != 0:
        raise RuntimeError('ffmpeg process video: {} error'.format(video_path))
        sys.exit(-1)

    sys.stdout.flush()
    return out_full_path


def mot_label():
    labels_map = {'person': 0}
    return labels_map


def visdrone_mcmot_label():
    labels_map = {
        'pedestrian': 0,
        'people': 1,
        'bicycle': 2,
        'car': 3,
        'van': 4,
        'truck': 5,
        'tricycle': 6,
        'awning-tricycle': 7,
        'bus': 8,
        'motor': 9,
    }
    return labels_map
