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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import time
from tqdm import tqdm

import numpy as np
import typing
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.static import InputSpec
from ppdet.optimizer import ModelEMA

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
from ppdet.utils.visualizer import visualize_results, save_result
from ppdet.metrics import get_infer_results, KeyPointTopDownCOCOEval, KeyPointTopDownCOCOWholeBadyHandEval, KeyPointTopDownMPIIEval, Pose3DEval
from ppdet.metrics import Metric, COCOMetric, VOCMetric, WiderFaceMetric, RBoxMetric, JDEDetMetric, SNIPERCOCOMetric, CULaneMetric
from ppdet.data.source.sniper_coco import SniperCOCODataSet
from ppdet.data.source.category import get_categories
import ppdet.utils.stats as stats
from ppdet.utils.fuse_utils import fuse_conv_bn
from ppdet.utils import profiler
from ppdet.modeling.post_process import multiclass_nms
from ppdet.modeling.lane_utils import imshow_lanes

from .callbacks import Callback, ComposeCallback, LogPrinter, Checkpointer, WiferFaceEval, VisualDLWriter, SniperProposalsGenerator, WandbCallback, SemiCheckpointer, SemiLogPrinter
from .export_utils import _dump_infer_config, _prune_input_spec, apply_to_static
from .naive_sync_bn import convert_syncbn

from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

from ppdet.utils.logger import setup_logger
logger = setup_logger('ppdet.engine')

__all__ = ['Trainer']
# MOT (Multiple Object Tracking) 
MOT_ARCH = ['JDE', 'FairMOT', 'DeepSORT', 'ByteTrack', 'CenterTrack']


class Trainer(object):
    def __init__(self, cfg_dict, mode='train'):
        self.cfg_dict = cfg_dict.copy()
        assert mode.lower() in ['train', 'eval', 'test'], \
                "mode should be 'train', 'eval' or 'test'"
        self.mode = mode.lower() #模式
        self.optimizer = None
        self.loaded_weights = False #是否以及加载好了权重
        self.use_amp = self.cfg_dict.get('amp', False) #不使用
        self.amp_level = self.cfg_dict.get('amp_level', 'O1')
        self.custom_white_list = self.cfg_dict.get('custom_white_list', None) #none
        self.custom_black_list = self.cfg_dict.get('custom_black_list', None) #none
        self.use_master_grad = self.cfg_dict.get('master_grad', False) #主节点的梯度
        if 'slim' in cfg_dict and cfg_dict['slim_type'] == 'PTQ':
            self.cfg_dict['TestDataset'] = create('TestDataset')()
        self._nranks = dist.get_world_size() #1
        self._local_rank = dist.get_rank() #0
        self.status = {}
        self.start_epoch = 0 
        self.end_epoch = 0 if 'epoch' not in self.cfg_dict else self.cfg_dict.epoch
        
        self._build_data_loader()
        self._build_model()
        self._eval_mode()
        self._print_params()
        self._optimizer_sync_bn_ema()
        # initial default callbacks
        self._init_callbacks()

        # initial default metrics
        self._init_metrics()
        self._reset_metrics()
      
      
        
    # build data loader
    def _build_data_loader(self):
        capital_mode = self.mode.capitalize() #首字母大写
        if self.cfg_dict.architecture in MOT_ARCH and self.mode in ['eval', 'test'] and self.cfg_dict.metric not in ['COCO', 'VOC']:
            self.dataset = self.cfg_dict['{}MOTDataset'.format(capital_mode)] = create('{}MOTDataset'.format(capital_mode))()
        else:
            self.dataset = self.cfg_dict['{}Dataset'.format(capital_mode)] = create('{}Dataset'.format(capital_mode))() #生成一个类

        if self.cfg_dict.architecture == 'DeepSORT' and self.mode == 'train':
            logger.error('DeepSORT has no need of training on mot dataset.')
            sys.exit(1)

        if self.cfg_dict.architecture == 'FairMOT' and self.mode == 'eval':
            images = self.parse_mot_images(self.cfg_dict)
            self.dataset.set_images(images)


        if self.mode == 'train':
            self.loader = create('{}Reader'.format(capital_mode))(
                self.dataset, self.cfg_dict.worker_num)
            
            
        if self.cfg_dict.architecture == 'JDE' and self.mode == 'train':
            self.cfg_dict['JDEEmbeddingHead'][
                'num_identities'] = self.dataset.seqs_reid_sum[0]
            # JDE only support single class MOT now.

        if self.cfg_dict.architecture == 'FairMOT' and self.mode == 'train':
            self.cfg_dict['FairMOTEmbeddingHead'][
                'seqs_reid_sum'] = self.dataset.seqs_reid_sum
            # FairMOT support single class and multi-class MOT now.
            



    def _build_model(self):
        # build model
        if 'model' not in self.cfg_dict:
            self.model = create(self.cfg_dict.architecture) #根据模型结构构建整个模型
        else:
            self.model = self.cfg_dict.model
            self.loaded_weights = True

        if self.cfg_dict.architecture == 'YOLOX':
            for k, m in self.model.named_sublayers():
                if isinstance(m, nn.BatchNorm2D):
                    m._epsilon = 1e-3  # for amp(fp16)
                    m._momentum = 0.97  # 0.03 in pytorch

        #normalize params for deploy
        if 'slim' in self.cfg_dict and self.cfg_dict['slim_type'] == 'OFA':
            self.model.model.load_meanstd(self.cfg_dict['TestReader'][
                'sample_transforms'])
        elif 'slim' in self.cfg_dict and self.cfg_dict['slim_type'] == 'Distill':
            self.model.student_model.load_meanstd(self.cfg_dict['TestReader'][
                'sample_transforms'])
        elif 'slim' in self.cfg_dict and self.cfg_dict[
                'slim_type'] == 'DistillPrune' and self.mode == 'train':
            self.model.student_model.load_meanstd(self.cfg_dict['TestReader'][
                'sample_transforms'])
        else:
            self.model.load_meanstd(self.cfg_dict['TestReader']['sample_transforms'])
            
    def _eval_mode(self):
        # EvalDataset build with BatchSampler to evaluate in single device
        # TODO: multi-device evaluate
        if self.mode == 'eval':
            if self.cfg_dict.architecture == 'FairMOT':
                self.loader = create('EvalMOTReader')(self.dataset, 0)
            elif self.cfg_dict.architecture == "METRO_Body":
                reader_name = '{}Reader'.format(self.mode.capitalize())
                self.loader = create(reader_name)(self.dataset, self.cfg_dict.worker_num)
            else:
                self._eval_batch_sampler = paddle.io.BatchSampler(
                    self.dataset, batch_size=self.cfg_dict.EvalReader['batch_size'])
                reader_name = '{}Reader'.format(self.mode.capitalize())
                # If metric is VOC, need to be set collate_batch=False.
                if self.cfg_dict.metric == 'VOC':
                    self.cfg_dict[reader_name]['collate_batch'] = False
                self.loader = create(reader_name)(self.dataset, self.cfg_dict.worker_num,
                                                  self._eval_batch_sampler)
        # TestDataset build after user set images, skip loader creation here
    def _print_params(self):
        # get Params
        print_params = self.cfg_dict.get('print_params', False)
        if print_params:
            params = sum([
                p.numel() for n, p in self.model.named_parameters()
                if all([x not in n for x in ['_mean', '_variance', 'aux_']])
            ])  # exclude BatchNorm running status
            logger.info('Model Params : {} M.'.format((params / 1e6).numpy()[
                0]))
            
            
    def _optimizer_sync_bn_ema(self):
        # build_optimizer
        if self.mode == 'train':
            steps_per_epoch = len(self.loader)
            if steps_per_epoch < 1:
                logger.warning(
                    "Samples in dataset are less than batch_size, please set smaller batch_size in TrainReader."
                )
            self.lr = create('LearningRate')(steps_per_epoch)
            self.optimizer = create('OptimizerBuilder')(self.lr, self.model)

            # Unstructured pruner is only enabled in the train mode.
            if self.cfg_dict.get('unstructured_prune'):
                self.pruner = create('UnstructuredPruner')(self.model,
                                                           steps_per_epoch)
        if self.use_amp and self.amp_level == 'O2':
            paddle_version = paddle.__version__[:3]
            # paddle version >= 2.5.0 or develop
            if paddle_version in ["2.5", "0.0"]:
                self.model, self.optimizer = paddle.amp.decorate(
                    models=self.model,
                    optimizers=self.optimizer,
                    level=self.amp_level,
                    master_grad=self.use_master_grad)
            else:
                self.model, self.optimizer = paddle.amp.decorate(
                    models=self.model,
                    optimizers=self.optimizer,
                    level=self.amp_level)

        # support sync_bn for npu/xpu
        if (paddle.get_device()[:3]=='npu' or paddle.get_device()[:3]=='xpu'):
            use_npu = ('use_npu' in self.cfg_dict and self.cfg_dict['use_npu'])
            use_xpu = ('use_xpu' in self.cfg_dict and self.cfg_dict['use_xpu'])
            norm_type = ('norm_type' in self.cfg_dict and self.cfg_dict['norm_type'])
            if norm_type == 'sync_bn' and (use_npu or use_xpu) and dist.get_world_size() > 1:
                convert_syncbn(self.model)

        self.use_ema = ('use_ema' in self.cfg_dict and self.cfg_dict['use_ema'])
        if self.use_ema:
            ema_decay = self.cfg_dict.get('ema_decay', 0.9998)
            ema_decay_type = self.cfg_dict.get('ema_decay_type', 'threshold')
            cycle_epoch = self.cfg_dict.get('cycle_epoch', -1)
            ema_black_list = self.cfg_dict.get('ema_black_list', None)
            ema_filter_no_grad = self.cfg_dict.get('ema_filter_no_grad', False)
            self.ema = ModelEMA(
                self.model,
                decay=ema_decay,
                ema_decay_type=ema_decay_type,
                cycle_epoch=cycle_epoch,
                ema_black_list=ema_black_list,
                ema_filter_no_grad=ema_filter_no_grad)





    def _init_callbacks(self):
        if self.mode == 'train':
            if self.cfg_dict.get('ssod_method',False) and self.cfg_dict['ssod_method'] == 'Semi_RTDETR':
                self._callbacks = [SemiLogPrinter(self), SemiCheckpointer(self)]
            else:
                self._callbacks = [LogPrinter(self), Checkpointer(self)]
            if self.cfg_dict.get('use_vdl', False):
                self._callbacks.append(VisualDLWriter(self))
            if self.cfg_dict.get('save_proposals', False):
                self._callbacks.append(SniperProposalsGenerator(self))
            if self.cfg_dict.get('use_wandb', False) or 'wandb' in self.cfg_dict:
                self._callbacks.append(WandbCallback(self))
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'eval':
            self._callbacks = [LogPrinter(self)]
            if self.cfg_dict.metric == 'WiderFace':
                self._callbacks.append(WiferFaceEval(self))
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'test' and self.cfg_dict.get('use_vdl', False):
            self._callbacks = [VisualDLWriter(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        else:
            self._callbacks = []
            self._compose_callback = None

    def _init_metrics(self, validate=False):
        if self.mode == 'test' or (self.mode == 'train' and not validate):
            self._metrics = []
            return
        classwise = self.cfg_dict['classwise'] if 'classwise' in self.cfg_dict else False
        if self.cfg_dict.metric == 'COCO' or self.cfg_dict.metric == "SNIPERCOCO":
            # TODO: bias should be unified
            bias = 1 if self.cfg_dict.get('bias', False) else 0
            output_eval = self.cfg_dict['output_eval'] \
                if 'output_eval' in self.cfg_dict else None
            save_prediction_only = self.cfg_dict.get('save_prediction_only', False)

            # pass clsid2catid info to metric instance to avoid multiple loading
            # annotation file
            clsid2catid = {v: k for k, v in self.dataset.catid2clsid.items()} \
                                if self.mode == 'eval' else None

            save_threshold = self.cfg_dict.get('save_threshold', 0)

            # when do validation in train, annotation file should be get from
            # EvalReader instead of self.dataset(which is TrainReader)
            if self.mode == 'train' and validate:
                eval_dataset = self.cfg_dict['EvalDataset']
                eval_dataset.check_or_download_dataset()
                anno_file = eval_dataset.get_anno()
                dataset = eval_dataset
            else:
                dataset = self.dataset
                anno_file = dataset.get_anno()

            IouType = self.cfg_dict['IouType'] if 'IouType' in self.cfg_dict else 'bbox'
            if self.cfg_dict.metric == "COCO":
                self._metrics = [
                    COCOMetric(
                        anno_file=anno_file,
                        clsid2catid=clsid2catid,
                        classwise=classwise,
                        output_eval=output_eval,
                        bias=bias,
                        IouType=IouType,
                        save_prediction_only=save_prediction_only,
                        save_threshold=save_threshold)
                ]
            elif self.cfg_dict.metric == "SNIPERCOCO":  # sniper
                self._metrics = [
                    SNIPERCOCOMetric(
                        anno_file=anno_file,
                        dataset=dataset,
                        clsid2catid=clsid2catid,
                        classwise=classwise,
                        output_eval=output_eval,
                        bias=bias,
                        IouType=IouType,
                        save_prediction_only=save_prediction_only)
                ]
        elif self.cfg_dict.metric == 'RBOX':
            # TODO: bias should be unified
            bias = self.cfg_dict['bias'] if 'bias' in self.cfg_dict else 0
            output_eval = self.cfg_dict['output_eval'] \
                if 'output_eval' in self.cfg_dict else None
            save_prediction_only = self.cfg_dict.get('save_prediction_only', False)
            imid2path = self.cfg_dict.get('imid2path', None)

            # when do validation in train, annotation file should be get from
            # EvalReader instead of self.dataset(which is TrainReader)
            anno_file = self.dataset.get_anno()
            if self.mode == 'train' and validate:
                eval_dataset = self.cfg_dict['EvalDataset']
                eval_dataset.check_or_download_dataset()
                anno_file = eval_dataset.get_anno()

            self._metrics = [
                RBoxMetric(
                    anno_file=anno_file,
                    classwise=classwise,
                    output_eval=output_eval,
                    bias=bias,
                    save_prediction_only=save_prediction_only,
                    imid2path=imid2path)
            ]
        elif self.cfg_dict.metric == 'VOC':
            output_eval = self.cfg_dict['output_eval'] \
                if 'output_eval' in self.cfg_dict else None
            save_prediction_only = self.cfg_dict.get('save_prediction_only', False)

            self._metrics = [
                VOCMetric(
                    label_list=self.dataset.get_label_list(),
                    class_num=self.cfg_dict.num_classes,
                    map_type=self.cfg_dict.map_type,
                    classwise=classwise,
                    output_eval=output_eval,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg_dict.metric == 'WiderFace':
            multi_scale = self.cfg_dict.multi_scale_eval if 'multi_scale_eval' in self.cfg_dict else True
            self._metrics = [
                WiderFaceMetric(
                    image_dir=os.path.join(self.dataset.dataset_dir,
                                           self.dataset.image_dir),
                    anno_file=self.dataset.get_anno(),
                    multi_scale=multi_scale)
            ]
        elif self.cfg_dict.metric == 'KeyPointTopDownCOCOEval':
            eval_dataset = self.cfg_dict['EvalDataset']
            eval_dataset.check_or_download_dataset()
            anno_file = eval_dataset.get_anno()
            save_prediction_only = self.cfg_dict.get('save_prediction_only', False)
            self._metrics = [
                KeyPointTopDownCOCOEval(
                    anno_file,
                    len(eval_dataset),
                    self.cfg_dict.num_joints,
                    self.cfg_dict.save_dir,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg_dict.metric == 'KeyPointTopDownCOCOWholeBadyHandEval':
            eval_dataset = self.cfg_dict['EvalDataset']
            eval_dataset.check_or_download_dataset()
            anno_file = eval_dataset.get_anno()
            save_prediction_only = self.cfg_dict.get('save_prediction_only', False)
            self._metrics = [
                KeyPointTopDownCOCOWholeBadyHandEval(
                    anno_file,
                    len(eval_dataset),
                    self.cfg_dict.num_joints,
                    self.cfg_dict.save_dir,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg_dict.metric == 'KeyPointTopDownMPIIEval':
            eval_dataset = self.cfg_dict['EvalDataset']
            eval_dataset.check_or_download_dataset()
            anno_file = eval_dataset.get_anno()
            save_prediction_only = self.cfg_dict.get('save_prediction_only', False)
            self._metrics = [
                KeyPointTopDownMPIIEval(
                    anno_file,
                    len(eval_dataset),
                    self.cfg_dict.num_joints,
                    self.cfg_dict.save_dir,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg_dict.metric == 'Pose3DEval':
            save_prediction_only = self.cfg_dict.get('save_prediction_only', False)
            self._metrics = [
                Pose3DEval(
                    self.cfg_dict.save_dir,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg_dict.metric == 'MOTDet':
            self._metrics = [JDEDetMetric(), ]
        elif self.cfg_dict.metric == 'CULaneMetric':
            output_eval = self.cfg_dict.get('output_eval', None)
            self._metrics = [
                CULaneMetric(
                    cfg_dict=self.cfg_dict,
                    output_eval=output_eval,
                    split=self.dataset.split,
                    dataset_dir=self.cfg_dict.dataset_dir)
            ]
        else:
            logger.warning("Metric not support for metric type {}".format(
                self.cfg_dict.metric))
            self._metrics = []

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def register_callbacks(self, callbacks):
        callbacks = [c for c in list(callbacks) if c is not None]
        for c in callbacks:
            assert isinstance(c, Callback), \
                    "metrics shoule be instances of subclass of Metric"
        self._callbacks.extend(callbacks)
        self._compose_callback = ComposeCallback(self._callbacks)

    def register_metrics(self, metrics):
        metrics = [m for m in list(metrics) if m is not None]
        for m in metrics:
            assert isinstance(m, Metric), \
                    "metrics shoule be instances of subclass of Metric"
        self._metrics.extend(metrics)

    def load_weights(self, weights, ARSL_eval=False):
        if self.loaded_weights:
            return
        self.start_epoch = 0 #重置开始训练轮次
        load_pretrain_weight(self.model, weights, ARSL_eval) #模型 权重(网址) 评估=false
        logger.debug("Load weights {} to start training".format(weights))

    def load_weights_sde(self, det_weights, reid_weights):
        if self.model.detector:
            load_weight(self.model.detector, det_weights)
            if self.model.reid:
                load_weight(self.model.reid, reid_weights)
        else:
            load_weight(self.model.reid, reid_weights)

    def resume_weights(self, weights):
        # support Distill resume weights
        if hasattr(self.model, 'student_model'):
            self.start_epoch = load_weight(self.model.student_model, weights,
                                           self.optimizer)
        else:
            self.start_epoch = load_weight(self.model, weights, self.optimizer,
                                           self.ema if self.use_ema else None)
        logger.debug("Resume weights of epoch {}".format(self.start_epoch))

    def prepare_model_and_data(self, validate):
        """准备模型和数据集，包括验证数据集的初始化和模型的静态图转化"""
        # 初始化验证数据集
        if validate:
            self.cfg_dict['EvalDataset'] = self.cfg_dict.EvalDataset = create("EvalDataset")()
            
        model = self.model
        # 将模型转化为静态图（如果需要）
        if self.cfg_dict.get('to_static', False):
            model = apply_to_static(self.cfg_dict, model)
        return model

    def apply_training_modes(self,model):
        """设置训练相关模式，包括同步批归一化和自动混合精度（AMP）"""
        # 同步批归一化（SyncBN）
        sync_bn = (getattr(self.cfg_dict, 'norm_type', None) == 'sync_bn' and
                (self.cfg_dict.use_gpu or self.cfg_dict.use_mlu) and self._nranks > 1)
        if sync_bn:
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # 启用自动混合精度（AMP）
        if self.use_amp:
            self.scaler = paddle.amp.GradScaler(
                enable=self.cfg_dict.use_gpu or self.cfg_dict.use_npu or self.cfg_dict.use_mlu,
                init_loss_scaling=self.cfg_dict.get('init_loss_scaling', 1024)
            )
            
        self.use_fused_allreduce_gradients = self.cfg_dict['use_fused_allreduce_gradients'] if 'use_fused_allreduce_gradients' in self.cfg_dict else False
        return model

    def setup_distributed_training(self,model):
        """配置分布式训练模型和优化器"""
        if self.cfg_dict.get('fleet', False):
            model = fleet.distributed_model(model)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
        elif self._nranks > 1:
            find_unused_parameters = self.cfg_dict.get('find_unused_parameters', False)
            model = paddle.DataParallel(model, find_unused_parameters=find_unused_parameters)
        return model

    def update_training_status_and_log(self):
        """更新训练状态信息，并进行相关日志记录与性能分析"""
        # 更新训练状态
        self.status.update({
            'epoch_id': self.start_epoch,
            'step_id': 0,
            'steps_per_epoch': len(self.loader)
        })
        self.status['batch_time'] = stats.SmoothedValue(self.cfg_dict.log_iter, fmt='{avg:.4f}')
        self.status['data_time'] = stats.SmoothedValue(self.cfg_dict.log_iter, fmt='{avg:.4f}')
        self.status['training_staus'] = stats.TrainingStats(self.cfg_dict.log_iter)

        # 打印Flops（如果需要）
        if self.cfg_dict.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg_dict.worker_num)
            self._flops(flops_loader)

        # 启动性能分析器（如果配置了）
        self.profiler_options = self.cfg_dict.get('profiler_options', None)
        if self.profiler_options:
            self._start_profiler(self.profiler_options)

        # 5. 初始化回调
        self._compose_callback.on_train_begin(self.status)

    def prepare_training(self, validate=False):
        """将所有函数组合起来，准备训练"""
        # 1. 准备模型和数据集
        model = self.prepare_model_and_data(validate)

        # 2. 应用训练模式设置（如同步批归一化、自动混合精度等）
        model = self.apply_training_modes(model)

        # 3. 设置分布式训练模式
        model = self.setup_distributed_training(model)

        # 4. 更新训练状态和日志
        self.update_training_status_and_log()

        return model

    def _train_batch_preprocess(self,data,iter_tic,step_id,epoch_id):
        def deep_pin(blob, blocking):
            if isinstance(blob, paddle.Tensor):
                return blob.cuda(blocking=blocking)
            elif isinstance(blob, dict):
                return {k: deep_pin(v, blocking) for k, v in blob.items()}
            elif isinstance(blob, (list, tuple)):
                return type(blob)([deep_pin(x, blocking) for x in blob])
            else:
                return blob
        if paddle.base.core.is_compiled_with_cuda():
            data = deep_pin(data, False)

        self.status['data_time'].update(time.time() - iter_tic) #数据处理时间
        self.status['step_id'] = step_id #第几步了
        profiler.add_profiler_step(self.profiler_options)
        self._compose_callback.on_step_begin(self.status) #回调函数
        data['epoch_id'] = epoch_id #数据字典
        if self.cfg_dict.get('to_static',False) and 'image_file' in data.keys():
            data.pop('image_file')
              
              
    def _train_batch_postprocess(self,outputs,iter_tic):
            curr_lr = self.optimizer.get_lr()
            self.lr.step()
            if self.cfg_dict.get('unstructured_prune'):
                self.pruner.step()
            self.optimizer.clear_grad()
            self.status['learning_rate'] = curr_lr

            if self._nranks < 2 or self._local_rank == 0:
                self.status['training_staus'].update(outputs)

            self.status['batch_time'].update(time.time() - iter_tic)
            self._compose_callback.on_step_end(self.status)
            if self.use_ema:
                self.ema.update()
            iter_tic = time.time()
      
    def _train_epoch(self,epoch_id,model):

        
        iter_tic = time.time()
        for step_id, data in enumerate(self.loader): #batch_size = 2
            
            self._train_batch_preprocess(data,iter_tic,step_id,epoch_id)


            if self.use_amp: #自动混合精度
                if isinstance(model, paddle.DataParallel) and self.use_fused_allreduce_gradients:
                    with model.no_sync():
                        with paddle.amp.auto_cast(
                                enable=self.cfg_dict.use_gpu or
                                self.cfg_dict.use_npu or self.cfg_dict.use_mlu,
                                custom_white_list=self.custom_white_list,
                                custom_black_list=self.custom_black_list,
                                level=self.amp_level):
                            # model forward
                            outputs = model(data)
                            loss = outputs['loss']
                        # model backward
                        scaled_loss = self.scaler.scale(loss)
                        scaled_loss.backward()
                    fused_allreduce_gradients(
                        list(model.parameters()), None)
                else:
                    with paddle.amp.auto_cast(
                            enable=self.cfg_dict.use_gpu or self.cfg_dict.use_npu or
                            self.cfg_dict.use_mlu,
                            custom_white_list=self.custom_white_list,
                            custom_black_list=self.custom_black_list,
                            level=self.amp_level):
                        # model forward
                        outputs = model(data)
                        loss = outputs['loss']
                    # model backward
                    scaled_loss = self.scaler.scale(loss)
                    scaled_loss.backward()
                # in dygraph mode, optimizer.minimize is equal to optimizer.step
                self.scaler.minimize(self.optimizer, scaled_loss)
            else:
                if isinstance(model, paddle.DataParallel) and self.use_fused_allreduce_gradients:
                    with model.no_sync():
                        # model forward
                        outputs = model(data)
                        loss = outputs['loss']
                        # model backward
                        loss.backward()
                    fused_allreduce_gradients(
                        list(model.parameters()), None)
                else:
                    # model forward
                    #逐批次训练
                    outputs = model(data) #image和真实都进来了
                    loss = outputs['loss']
                    # model backward
                    loss.backward()
                self.optimizer.step() #参数更新
                
                
                
            self._train_batch_postprocess(outputs,iter_tic)
            

    def _train_epoch_prepare(self,epoch_id):
        self.status['mode'] = 'train'
        self.status['epoch_id'] = epoch_id
        self._compose_callback.on_epoch_begin(self.status)
        self.loader.dataset.set_epoch(epoch_id)



    def handle_pruning_and_snapshot(self, epoch_id):
        """Handle pruning and determine if snapshot should be taken."""
        # Handle unstructured pruning if enabled
        if self.cfg_dict.get('unstructured_prune'):
            self.pruner.update_params()

        # Determine snapshot conditions
        is_primary = self._nranks < 2 or self._local_rank == 0
        is_pose_eval = self.cfg_dict.metric == "Pose3DEval"
        is_save_epoch = (epoch_id + 1) % self.cfg_dict.snapshot_epoch == 0 or epoch_id == self.end_epoch - 1
        is_snapshot = (is_primary or is_pose_eval) and is_save_epoch

        # Apply EMA weights if snapshot and EMA are enabled
        weight = None
        if is_snapshot and self.use_ema:
            weight = copy.deepcopy(self.model.state_dict())
            self.model.set_dict(self.ema.apply())
            self.status['weight'] = weight

        return is_snapshot, weight #快照，权重


    def perform_validate(self, validate, is_snapshot):
        """Initialize evaluation loader and perform validation if required."""
        if validate and is_snapshot:
            # _eval_loader
            if not hasattr(self, '_eval_loader'):
                self._eval_dataset = self.cfg_dict.EvalDataset #验证数据集
                self._eval_batch_sampler = paddle.io.BatchSampler(
                    self._eval_dataset, batch_size=self.cfg_dict.EvalReader['batch_size']
                )
                # Handle VOC and Pose3DEval-specific configurations
                if self.cfg_dict.metric == 'VOC':
                    self.cfg_dict['EvalReader']['collate_batch'] = False
                if self.cfg_dict.metric == "Pose3DEval":
                    self._eval_loader = create('EvalReader')(
                        self._eval_dataset, self.cfg_dict.worker_num
                    )
                else:
                    self._eval_loader = create('EvalReader')(
                        self._eval_dataset, self.cfg_dict.worker_num, batch_sampler=self._eval_batch_sampler
                    )

            # _metrics_initialized
            if validate and not hasattr(self, '_metrics_initialized'):
                self._metrics_initialized = True
                self._init_metrics(validate=validate)
                self._reset_metrics()

            # Perform evaluation
            with paddle.no_grad():
                self.status['save_best_model'] = True
                self._eval_with_loader(self._eval_loader) #保存最好的模型计算指标


    def reset_weights(self, weight):
        """Reset the original weights after snapshot."""
        if weight:
            self.model.set_dict(weight)
            self.status.pop('weight')


    def _end_epoch(self, epoch_id, validate):
        """Main training step logic."""
        # 判断
        is_snapshot, weight = self.handle_pruning_and_snapshot(epoch_id) #保存快照 权重

        # 保存权重
        self._compose_callback.on_epoch_end(self.status) #

        # 验证
        self.perform_validate(validate, is_snapshot) #true true

        #重置
        self.reset_weights(weight)


        
    def train(self, validate=False):
        assert self.mode == 'train', "Model not in 'train' mode"
        model = self.prepare_training(validate)
        


    

        for epoch_id in range(self.start_epoch, self.cfg_dict.epoch): #0  72
            
            self._train_epoch_prepare(epoch_id)
            model.train()
            self._train_epoch(epoch_id,model)
            self._end_epoch(epoch_id, validate)


        self._compose_callback.on_train_end(self.status) #总的训练结束

    def _eval_with_loader(self, loader):
        sample_num = 0
        tic = time.time()
        self._compose_callback.on_epoch_begin(self.status)
        self.status['mode'] = 'eval'

        self.model.eval()
        if self.cfg_dict.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg_dict.worker_num, self._eval_batch_sampler)
            self._flops(flops_loader)
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            # forward
            if self.use_amp:
                with paddle.amp.auto_cast(
                        enable=self.cfg_dict.use_gpu or self.cfg_dict.use_npu or
                        self.cfg_dict.use_mlu,
                        custom_white_list=self.custom_white_list,
                        custom_black_list=self.custom_black_list,
                        level=self.amp_level):
                    outs = self.model(data)
            else:
                outs = self.model(data)

            # update metrics
            for metric in self._metrics:
                metric.update(data, outs)

            # multi-scale inputs: all inputs have same im_id
            if isinstance(data, typing.Sequence):
                sample_num += data[0]['im_id'].numpy().shape[0]
            else:
                sample_num += data['im_id'].numpy().shape[0]
            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        self._compose_callback.on_epoch_end(self.status)
        # reset metric states for metric may performed multiple times
        self._reset_metrics()

    def evaluate(self):
        # get distributed model
        if self.cfg_dict.get('fleet', False):
            self.model = fleet.distributed_model(self.model)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
        elif self._nranks > 1:
            find_unused_parameters = self.cfg_dict[
                'find_unused_parameters'] if 'find_unused_parameters' in self.cfg_dict else False
            self.model = paddle.DataParallel(
                self.model, find_unused_parameters=find_unused_parameters)
        with paddle.no_grad():
            self._eval_with_loader(self.loader)

    def _eval_with_loader_slice(self,
                                loader,
                                slice_size=[640, 640],
                                overlap_ratio=[0.25, 0.25],
                                combine_method='nms',
                                match_threshold=0.6,
                                match_metric='iou'):
        sample_num = 0
        tic = time.time()
        self._compose_callback.on_epoch_begin(self.status)
        self.status['mode'] = 'eval'
        self.model.eval()
        if self.cfg_dict.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg_dict.worker_num, self._eval_batch_sampler)
            self._flops(flops_loader)

        merged_bboxs = []
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            # forward
            if self.use_amp:
                with paddle.amp.auto_cast(
                        enable=self.cfg_dict.use_gpu or self.cfg_dict.use_npu or
                        self.cfg_dict.use_mlu,
                        custom_white_list=self.custom_white_list,
                        custom_black_list=self.custom_black_list,
                        level=self.amp_level):
                    outs = self.model(data)
            else:
                outs = self.model(data)

            shift_amount = data['st_pix']
            outs['bbox'][:, 2:4] = outs['bbox'][:, 2:4] + shift_amount
            outs['bbox'][:, 4:6] = outs['bbox'][:, 4:6] + shift_amount
            merged_bboxs.append(outs['bbox'])

            if data['is_last'] > 0:
                # merge matching predictions
                merged_results = {'bbox': []}
                if combine_method == 'nms':
                    final_boxes = multiclass_nms(
                        np.concatenate(merged_bboxs), self.cfg_dict.num_classes,
                        match_threshold, match_metric)
                    merged_results['bbox'] = np.concatenate(final_boxes)
                elif combine_method == 'concat':
                    merged_results['bbox'] = np.concatenate(merged_bboxs)
                else:
                    raise ValueError(
                        "Now only support 'nms' or 'concat' to fuse detection results."
                    )
                merged_results['im_id'] = np.array([[0]])
                merged_results['bbox_num'] = np.array(
                    [len(merged_results['bbox'])])

                merged_bboxs = []
                data['im_id'] = data['ori_im_id']
                # update metrics
                for metric in self._metrics:
                    metric.update(data, merged_results)

                # multi-scale inputs: all inputs have same im_id
                if isinstance(data, typing.Sequence):
                    sample_num += data[0]['im_id'].numpy().shape[0]
                else:
                    sample_num += data['im_id'].numpy().shape[0]

            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        self._compose_callback.on_epoch_end(self.status)
        # reset metric states for metric may performed multiple times
        self._reset_metrics()

    def evaluate_slice(self,
                       slice_size=[640, 640],
                       overlap_ratio=[0.25, 0.25],
                       combine_method='nms',
                       match_threshold=0.6,
                       match_metric='iou'):
        with paddle.no_grad():
            self._eval_with_loader_slice(self.loader, slice_size, overlap_ratio,
                                         combine_method, match_threshold,
                                         match_metric)

    def slice_predict(self,
                      images,
                      slice_size=[640, 640],
                      overlap_ratio=[0.25, 0.25],
                      combine_method='nms',
                      match_threshold=0.6,
                      match_metric='iou',
                      draw_threshold=0.5,
                      output_dir='output',
                      save_results=False,
                      visualize=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_slice_images(images, slice_size, overlap_ratio)
        loader = create('TestReader')(self.dataset, 0)
        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            # mem
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg_dict[
                'save_prediction_only'] if 'save_prediction_only' in self.cfg_dict else None
            output_eval = self.cfg_dict[
                'output_eval'] if 'output_eval' in self.cfg_dict else None

            # modify
            self.mode = '_test'
            self.cfg_dict['save_prediction_only'] = True
            self.cfg_dict['output_eval'] = output_dir
            self.cfg_dict['imid2path'] = imid2path
            self._init_metrics()

            # restore
            self.mode = mode
            self.cfg_dict.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg_dict['save_prediction_only'] = save_prediction_only

            self.cfg_dict.pop('output_eval')
            if output_eval is not None:
                self.cfg_dict['output_eval'] = output_eval

            self.cfg_dict.pop('imid2path')

            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg_dict.metric, anno_file=anno_file)

        # Run Infer
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg_dict.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)

        results = []  # all images
        merged_bboxs = []  # single image
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            outs = self.model(data)

            outs['bbox'] = outs['bbox'].numpy()  # only in test mode
            shift_amount = data['st_pix']
            outs['bbox'][:, 2:4] = outs['bbox'][:, 2:4] + shift_amount.numpy()
            outs['bbox'][:, 4:6] = outs['bbox'][:, 4:6] + shift_amount.numpy()
            merged_bboxs.append(outs['bbox'])

            if data['is_last'] > 0:
                # merge matching predictions
                merged_results = {'bbox': []}
                if combine_method == 'nms':
                    final_boxes = multiclass_nms(
                        np.concatenate(merged_bboxs), self.cfg_dict.num_classes,
                        match_threshold, match_metric)
                    merged_results['bbox'] = np.concatenate(final_boxes)
                elif combine_method == 'concat':
                    merged_results['bbox'] = np.concatenate(merged_bboxs)
                else:
                    raise ValueError(
                        "Now only support 'nms' or 'concat' to fuse detection results."
                    )
                merged_results['im_id'] = np.array([[0]])
                merged_results['bbox_num'] = np.array(
                    [len(merged_results['bbox'])])

                merged_bboxs = []
                data['im_id'] = data['ori_im_id']

                for _m in metrics:
                    _m.update(data, merged_results)

                for key in ['im_shape', 'scale_factor', 'im_id']:
                    if isinstance(data, typing.Sequence):
                        merged_results[key] = data[0][key]
                    else:
                        merged_results[key] = data[key]
                for key, value in merged_results.items():
                    if hasattr(value, 'numpy'):
                        merged_results[key] = value.numpy()
                results.append(merged_results)

        for _m in metrics:
            _m.accumulate()
            _m.reset()

        if visualize:
            for outs in results:
                batch_res = get_infer_results(outs, clsid2catid)
                bbox_num = outs['bbox_num']

                start = 0
                for i, im_id in enumerate(outs['im_id']):
                    image_path = imid2path[int(im_id)]
                    image = Image.open(image_path).convert('RGB')
                    image = ImageOps.exif_transpose(image)
                    self.status['original_image'] = np.array(image.copy())

                    end = start + bbox_num[i]
                    bbox_res = batch_res['bbox'][start:end] \
                            if 'bbox' in batch_res else None
                    mask_res = batch_res['mask'][start:end] \
                            if 'mask' in batch_res else None
                    segm_res = batch_res['segm'][start:end] \
                            if 'segm' in batch_res else None
                    keypoint_res = batch_res['keypoint'][start:end] \
                            if 'keypoint' in batch_res else None
                    pose3d_res = batch_res['pose3d'][start:end] \
                            if 'pose3d' in batch_res else None
                    image = visualize_results(
                        image, bbox_res, mask_res, segm_res, keypoint_res,
                        pose3d_res, int(im_id), catid2name, draw_threshold)
                    self.status['result_image'] = np.array(image.copy())
                    if self._compose_callback:
                        self._compose_callback.on_step_end(self.status)
                    # save image with detection
                    save_name = self._get_save_image_name(output_dir,
                                                          image_path)
                    logger.info("Detection bbox results save in {}".format(
                        save_name))
                    image.save(save_name, quality=95)

                    start = end

    def predict(self,
                images,
                draw_threshold=0.5,
                output_dir='output',
                save_results=False,
                visualize=True,
                save_threshold=0,
                do_eval=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if do_eval:
            save_threshold = 0.0
        self.dataset.set_images(images, do_eval=do_eval)
        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            # mem
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg_dict[
                'save_prediction_only'] if 'save_prediction_only' in self.cfg_dict else None
            output_eval = self.cfg_dict[
                'output_eval'] if 'output_eval' in self.cfg_dict else None

            # modify
            self.mode = '_test'
            self.cfg_dict['save_prediction_only'] = True
            self.cfg_dict['output_eval'] = output_dir
            self.cfg_dict['imid2path'] = imid2path
            self.cfg_dict['save_threshold'] = save_threshold
            self._init_metrics()

            # restore
            self.mode = mode
            self.cfg_dict.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg_dict['save_prediction_only'] = save_prediction_only            

            self.cfg_dict.pop('output_eval')
            if output_eval is not None:
                self.cfg_dict['output_eval'] = output_eval

            self.cfg_dict.pop('imid2path')

            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg_dict.metric, anno_file=anno_file)

        # Run Infer
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg_dict.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)
        results = []
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            if hasattr(self.model, 'modelTeacher'):
                outs = self.model.modelTeacher(data)
            else:
                outs = self.model(data)
            for _m in metrics:
                _m.update(data, outs)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            results.append(outs)

        # sniper
        if type(self.dataset) == SniperCOCODataSet:
            results = self.dataset.anno_cropper.aggregate_chips_detections(
                results)

        for _m in metrics:
            _m.accumulate()
            _m.reset()

        if visualize:
            for outs in results:
                batch_res = get_infer_results(outs, clsid2catid)
                bbox_num = outs['bbox_num']

                start = 0
                for i, im_id in enumerate(outs['im_id']):
                    image_path = imid2path[int(im_id)]
                    image = Image.open(image_path).convert('RGB')
                    image = ImageOps.exif_transpose(image)
                    self.status['original_image'] = np.array(image.copy())

                    end = start + bbox_num[i]
                    bbox_res = batch_res['bbox'][start:end] \
                            if 'bbox' in batch_res else None
                    mask_res = batch_res['mask'][start:end] \
                            if 'mask' in batch_res else None
                    segm_res = batch_res['segm'][start:end] \
                            if 'segm' in batch_res else None
                    keypoint_res = batch_res['keypoint'][start:end] \
                            if 'keypoint' in batch_res else None
                    pose3d_res = batch_res['pose3d'][start:end] \
                            if 'pose3d' in batch_res else None
                    image = visualize_results(
                        image, bbox_res, mask_res, segm_res, keypoint_res,
                        pose3d_res, int(im_id), catid2name, draw_threshold)
                    self.status['result_image'] = np.array(image.copy())
                    if self._compose_callback:
                        self._compose_callback.on_step_end(self.status)
                    # save image with detection
                    save_name = self._get_save_image_name(output_dir,
                                                          image_path)
                    logger.info("Detection bbox results save in {}".format(
                        save_name))
                    image.save(save_name, quality=95)

                    start = end
        return results

    def _get_save_image_name(self, output_dir, image_path):
        """
        Get save image name from source image path.
        """
        image_name = os.path.split(image_path)[-1]
        name, ext = os.path.splitext(image_name)
        return os.path.join(output_dir, "{}".format(name)) + ext

    def _get_infer_cfg_and_input_spec(self,
                                      save_dir,
                                      prune_input=True,
                                      kl_quant=False,
                                      yaml_name=None):
        if yaml_name is None:
            yaml_name = 'infer_cfg.yml'
        image_shape = None
        im_shape = [None, 2]
        scale_factor = [None, 2]
        if self.cfg_dict.architecture in MOT_ARCH:
            test_reader_name = 'TestMOTReader'
        else:
            test_reader_name = 'TestReader'
        if 'inputs_def' in self.cfg_dict[test_reader_name]:
            inputs_def = self.cfg_dict[test_reader_name]['inputs_def']
            image_shape = inputs_def.get('image_shape', None)
        # set image_shape=[None, 3, -1, -1] as default
        if image_shape is None:
            image_shape = [None, 3, -1, -1]

        if len(image_shape) == 3:
            image_shape = [None] + image_shape
        else:
            im_shape = [image_shape[0], 2]
            scale_factor = [image_shape[0], 2]

        if hasattr(self.model, 'deploy'):
            self.model.deploy = True

        if 'slim' not in self.cfg_dict:
            for layer in self.model.sublayers():
                if hasattr(layer, 'convert_to_deploy'):
                    layer.convert_to_deploy()

        if hasattr(self.cfg_dict, 'export') and 'fuse_conv_bn' in self.cfg_dict[
                'export'] and self.cfg_dict['export']['fuse_conv_bn']:
            self.model = fuse_conv_bn(self.model)

        export_post_process = self.cfg_dict['export'].get(
            'post_process', False) if hasattr(self.cfg_dict, 'export') else True
        export_nms = self.cfg_dict['export'].get('nms', False) if hasattr(
            self.cfg_dict, 'export') else True
        export_benchmark = self.cfg_dict['export'].get(
            'benchmark', False) if hasattr(self.cfg_dict, 'export') else False
        if hasattr(self.model, 'fuse_norm'):
            self.model.fuse_norm = self.cfg_dict['TestReader'].get('fuse_normalize',
                                                              False)
        if hasattr(self.model, 'export_post_process'):
            self.model.export_post_process = export_post_process if not export_benchmark else False
        if hasattr(self.model, 'export_nms'):
            self.model.export_nms = export_nms if not export_benchmark else False
        if export_post_process and not export_benchmark:
            image_shape = [None] + image_shape[1:]

        # Save infer cfg_dict
        _dump_infer_config(self.cfg_dict,
                           os.path.join(save_dir, yaml_name), image_shape,
                           self.model)

        input_spec = [{
            "image": InputSpec(
                shape=image_shape, name='image'),
            "im_shape": InputSpec(
                shape=im_shape, name='im_shape'),
            "scale_factor": InputSpec(
                shape=scale_factor, name='scale_factor')
        }]
        if self.cfg_dict.architecture == 'DeepSORT':
            input_spec[0].update({
                "crops": InputSpec(
                    shape=[None, 3, 192, 64], name='crops')
            })

        if self.cfg_dict.architecture == 'CLRNet':
            input_spec[0].update({
                "full_img_path": str,
                "img_name": str,
            })
        if prune_input:
            static_model = paddle.jit.to_static(
                self.model, input_spec=input_spec, full_graph=True)
            # NOTE: dy2st do not pruned program, but jit.save will prune program
            # input spec, prune input spec here and save with pruned input spec
            pruned_input_spec = _prune_input_spec(
                input_spec, static_model.forward.main_program,
                static_model.forward.outputs)
        else:
            static_model = None
            pruned_input_spec = input_spec

        # TODO: Hard code, delete it when support prune input_spec.
        if self.cfg_dict.architecture == 'PicoDet' and not export_post_process:
            pruned_input_spec = [{
                "image": InputSpec(
                    shape=image_shape, name='image')
            }]
        if kl_quant:
            if self.cfg_dict.architecture == 'PicoDet' or 'ppyoloe' in self.cfg_dict.weights:
                pruned_input_spec = [{
                    "image": InputSpec(
                        shape=image_shape, name='image'),
                    "scale_factor": InputSpec(
                        shape=scale_factor, name='scale_factor')
                }]
            elif 'tinypose' in self.cfg_dict.weights:
                pruned_input_spec = [{
                    "image": InputSpec(
                        shape=image_shape, name='image')
                }]

        return static_model, pruned_input_spec

    def export(self, output_dir='output_inference', for_fd=False):
        if hasattr(self.model, 'aux_neck'):
            self.model.__delattr__('aux_neck')
        if hasattr(self.model, 'aux_head'):
            self.model.__delattr__('aux_head')
        self.model.eval()

        model_name = os.path.splitext(os.path.split(self.cfg_dict.filename)[-1])[0]
        if for_fd:
            save_dir = output_dir
            save_name = 'inference'
            yaml_name = 'inference.yml'
        else:
            save_dir = os.path.join(output_dir, model_name)
            save_name = 'model'
            yaml_name = None

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        static_model, pruned_input_spec = self._get_infer_cfg_and_input_spec(
            save_dir, yaml_name=yaml_name)

        # dy2st and save model
        if 'slim' not in self.cfg_dict or 'QAT' not in self.cfg_dict['slim_type']:
            paddle.jit.save(
                static_model,
                os.path.join(save_dir, save_name),
                input_spec=pruned_input_spec)
        else:
            self.cfg_dict.slim.save_quantized_model(
                self.model,
                os.path.join(save_dir, save_name),
                input_spec=pruned_input_spec)
        logger.info("Export model and saved in {}".format(save_dir))

    def post_quant(self, output_dir='output_inference'):
        model_name = os.path.splitext(os.path.split(self.cfg_dict.filename)[-1])[0]
        save_dir = os.path.join(output_dir, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx, data in enumerate(self.loader):
            self.model(data)
            if idx == int(self.cfg_dict.get('quant_batch_num', 10)):
                break

        # TODO: support prune input_spec
        kl_quant = True if hasattr(self.cfg_dict.slim, 'ptq') else False
        _, pruned_input_spec = self._get_infer_cfg_and_input_spec(
            save_dir, prune_input=False, kl_quant=kl_quant)

        self.cfg_dict.slim.save_quantized_model(
            self.model,
            os.path.join(save_dir, 'model'),
            input_spec=pruned_input_spec)
        logger.info("Export Post-Quant model and saved in {}".format(save_dir))

    def _flops(self, loader):
        if hasattr(self.model, 'aux_neck'):
            self.model.__delattr__('aux_neck')
        if hasattr(self.model, 'aux_head'):
            self.model.__delattr__('aux_head')
        self.model.eval()
        try:
            import paddleslim
        except Exception as e:
            logger.warning(
                'Unable to calculate flops, please install paddleslim, for example: `pip install paddleslim`'
            )
            return

        from paddleslim.analysis import dygraph_flops as flops
        input_data = None
        for data in loader:
            input_data = data
            break

        input_spec = [{
            "image": input_data['image'][0].unsqueeze(0),
            "im_shape": input_data['im_shape'][0].unsqueeze(0),
            "scale_factor": input_data['scale_factor'][0].unsqueeze(0)
        }]
        flops = flops(self.model, input_spec) / (1000**3)
        logger.info(" Model FLOPs : {:.6f}G. (image shape is {})".format(
            flops, input_data['image'][0].unsqueeze(0).shape))

    def parse_mot_images(self, cfg_dict):
        import glob
        # for quant
        dataset_dir = cfg_dict['EvalMOTDataset'].dataset_dir
        data_root = cfg_dict['EvalMOTDataset'].data_root
        data_root = '{}/{}'.format(dataset_dir, data_root)
        seqs = os.listdir(data_root)
        seqs.sort()
        all_images = []
        for seq in seqs:
            infer_dir = os.path.join(data_root, seq)
            assert infer_dir is None or os.path.isdir(infer_dir), \
                "{} is not a directory".format(infer_dir)
            images = set()
            exts = ['jpg', 'jpeg', 'png', 'bmp']
            exts += [ext.upper() for ext in exts]
            for ext in exts:
                images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
            images = list(images)
            images.sort()
            assert len(images) > 0, "no image found in {}".format(infer_dir)
            all_images.extend(images)
            logger.info("Found {} inference images in total.".format(
                len(images)))
        return all_images

    def predict_culane(self,
                       images,
                       output_dir='output',
                       save_results=False,
                       visualize=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            # mem
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg_dict[
                'save_prediction_only'] if 'save_prediction_only' in self.cfg_dict else None
            output_eval = self.cfg_dict[
                'output_eval'] if 'output_eval' in self.cfg_dict else None

            # modify
            self.mode = '_test'
            self.cfg_dict['save_prediction_only'] = True
            self.cfg_dict['output_eval'] = output_dir
            self.cfg_dict['imid2path'] = imid2path
            self._init_metrics()

            # restore
            self.mode = mode
            self.cfg_dict.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg_dict['save_prediction_only'] = save_prediction_only

            self.cfg_dict.pop('output_eval')
            if output_eval is not None:
                self.cfg_dict['output_eval'] = output_eval

            self.cfg_dict.pop('imid2path')

            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        # Run Infer
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg_dict.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)
        results = []
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            outs = self.model(data)

            for _m in metrics:
                _m.update(data, outs)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            results.append(outs)

        for _m in metrics:
            _m.accumulate()
            _m.reset()

        if visualize:
            import cv2

            for outs in results:
                for i in range(len(outs['img_path'])):
                    lanes = outs['lanes'][i]
                    img_path = outs['img_path'][i]
                    img = cv2.imread(img_path)
                    out_file = os.path.join(output_dir,
                                            os.path.basename(img_path))
                    lanes = [
                        lane.to_array(
                            sample_y_range=[
                                self.cfg_dict['sample_y']['start'],
                                self.cfg_dict['sample_y']['end'],
                                self.cfg_dict['sample_y']['step']
                            ],
                            img_w=self.cfg_dict.ori_img_w,
                            img_h=self.cfg_dict.ori_img_h) for lane in lanes
                    ]
                    imshow_lanes(img, lanes, out_file=out_file)

        return results
