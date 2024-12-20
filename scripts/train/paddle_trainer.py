# import os
# import sys
# # add python path of PaddleDetection to sys.path
# parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
# sys.path.insert(0, parent_path)
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer, TrainerCot, init_parallel_env, set_random_seed, init_fleet_env
from ppdet.engine.trainer_ssod import Trainer_DenseTeacher, Trainer_ARSL, Trainer_Semi_RTDETR
from ppdet.slim import build_slim_model
import ppdet.utils.check as check
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.utils.logger import setup_logger
logger = setup_logger('train')


import warnings
warnings.filterwarnings("ignore")

import paddle
import numpy as np
import random
seed = 42
# 设置随机数
paddle.seed(seed)
np.random.seed(seed)
random.seed(seed)


def parse_args():
    parser = ArgsParser()  # 参数解析器
    parser.add_argument("--eval", action='store_true', default=False, help="Whether to perform evaluation in train")
    parser.add_argument("-r", "--resume", default=None, help="weights path for resume")
    parser.add_argument("--slim_config", default=None, type=str, help="Configuration file of slim method.")
    parser.add_argument("--enable_ce", type=bool, default=False, help="If set True, enable continuous evaluation job. This flag is only used for internal paddle.")
    parser.add_argument("--amp", action='store_true', default=False, help="Enable auto mixed precision training.")
    parser.add_argument("--fleet", action='store_true', default=False, help="Use fleet or not")
    parser.add_argument("--use_vdl", type=bool, default=False, help="whether to record the data to VisualDL.")
    parser.add_argument('--vdl_log_dir', type=str, default="vdl_log_dir/scalar", help='VisualDL logging directory for scalar.')
    parser.add_argument("--use_wandb", type=bool, default=False, help="whether to record the data to wandb.")
    parser.add_argument('--save_prediction_only', action='store_true', default=False, help='Whether to save the evaluation results only')
    parser.add_argument('--profiler_options', type=str, default=None, help="The option of profiler, which should be in format \"key1=value1;key2=value2;key3=value3\". please see ppdet/utils/profiler.py for detail.")
    parser.add_argument('--save_proposals', action='store_true', default=False, help='Whether to save the train proposals')
    parser.add_argument('--proposals_path', type=str, default="sniper/proposals.json", help='Train proposals directory')
    parser.add_argument("--to_static", action='store_true', default=False, help="Enable dy2st to train.")

    args = parser.parse_args()
    return args



class PaddleDetectionTrainer:
    def __init__(self):
        self.FLAGS = parse_args() #主观上只传入整体的配置文件
        self.cfg_dict = load_config(self.FLAGS.config)
        self.cfg_dict = merge_args(self.cfg_dict, self.FLAGS) #融入手动传入的参数
        self.place = self._set_device()
        self._build_trainer()

    def _set_device(self):
        """Set the device based on configuration."""

        # Disable device flags (gpu, npu, xpu, mlu) in config by default
        for device in ['use_npu', 'use_xpu', 'use_gpu', 'use_mlu']:
            if device not in self.cfg_dict:
                setattr(self.cfg_dict, device, False)


        # Set device based on enabled flag priority (gpu > npu > xpu > mlu > cpu)
        device_priority = ['gpu', 'npu', 'xpu', 'mlu', 'cpu']
        for device in device_priority:
            if self.cfg_dict.get(f'use_{device}', False) or device == 'cpu':
                paddle.set_device(device)
                break

        # Apply slimming config if enabled
        if self.FLAGS.slim_config:
            self.cfg_dict = build_slim_model(self.cfg_dict, self.FLAGS.slim_config)

        # Configuration and environment checks
        for device in ['gpu', 'npu', 'xpu', 'mlu']:
            check_method = getattr(check, f'check_{device}', None)
            if check_method:
                check_method(self.cfg_dict.get(f'use_{device}', False))
        check.check_config(self.cfg_dict)
        check.check_version()
        
        
    def _build_trainer(self):
        """Initialize trainer based on the configuration."""
        ssod_method = self.cfg_dict.get('ssod_method') #SSOD 是 Semi-Supervised Object Detection 的缩写，即半监督目标检测。它是一种结合了有标签和无标签数据的目标检测方法
        if ssod_method == 'DenseTeacher':
            self.trainer = Trainer_DenseTeacher(self.cfg_dict, mode='train')
        elif ssod_method == 'ARSL':
            self.trainer = Trainer_ARSL(self.cfg_dict, mode='train')
        elif ssod_method == 'Semi_RTDETR':
            self.trainer = Trainer_Semi_RTDETR(self.cfg_dict, mode='train')
        elif self.cfg_dict.get('use_cot', False):
            self.trainer = TrainerCot(self.cfg_dict, mode='train')
        else:
            self.trainer = Trainer(self.cfg_dict, mode='train') #这里加载数据集和预训练权重

        # Load weights if specified
        if self.FLAGS.resume is not None:
            self.trainer.resume_weights(self.FLAGS.resume)
        elif 'pretrain_teacher_weights' in self.cfg_dict and 'pretrain_student_weights' in self.cfg_dict:
            self.trainer.load_semi_weights(
                self.cfg_dict['pretrain_teacher_weights'],
                self.cfg_dict['pretrain_student_weights']
            )
        elif 'pretrain_weights' in self.cfg_dict:
            self.trainer.load_weights(self.cfg_dict['pretrain_weights'])

    def train(self):
        """Run training based on the configuration."""
        self.trainer.train(self.FLAGS.eval)
