import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import paddle
    
from ppdet.core.workspace import load_config, merge_config

from ppdet.engine import Trainer, TrainerCot, init_parallel_env, set_random_seed, init_fleet_env
from ppdet.engine.trainer_ssod import Trainer_DenseTeacher, Trainer_ARSL, Trainer_Semi_RTDETR

from ppdet.slim import build_slim_model

from ppdet.utils.cli import ArgsParser, merge_args
import ppdet.utils.check as check
from ppdet.utils.logger import setup_logger
logger = setup_logger('train')


def parse_args():
    parser = ArgsParser() #参数解析器
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")
    parser.add_argument(
        "-r", "--resume", default=None, help="weights path for resume")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--enable_ce",
        type=bool,
        default=False,
        help="If set True, enable continuous evaluation job."
        "This flag is only used for internal test.")
    parser.add_argument(
        "--amp",
        action='store_true',
        default=False,
        help="Enable auto mixed precision training.")
    parser.add_argument(
        "--fleet", action='store_true', default=False, help="Use fleet or not")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/scalar",
        help='VisualDL logging directory for scalar.')
    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=False,
        help="whether to record the data to wandb.")
    parser.add_argument(
        '--save_prediction_only',
        action='store_true',
        default=False,
        help='Whether to save the evaluation results only')
    parser.add_argument(
        '--profiler_options',
        type=str,
        default=None,
        help="The option of profiler, which should be in "
        "format \"key1=value1;key2=value2;key3=value3\"."
        "please see ppdet/utils/profiler.py for detail.")
    parser.add_argument(
        '--save_proposals',
        action='store_true',
        default=False,
        help='Whether to save the train proposals')
    parser.add_argument(
        '--proposals_path',
        type=str,
        default="sniper/proposals.json",
        help='Train proposals directory')
    parser.add_argument(
        "--to_static",
        action='store_true',
        default=False,
        help="Enable dy2st to train.")

    args = parser.parse_args()
    return args


def run(FLAGS, cfg_dict):
    # init fleet environment
    if cfg_dict.fleet: #全局参数字典键值对
        init_fleet_env(cfg_dict.get('find_unused_parameters', False))
    else:
        # init parallel environment if nranks > 1
        init_parallel_env()

    if FLAGS.enable_ce:
        set_random_seed(0)

    # build trainer
    ssod_method = cfg_dict.get('ssod_method', None) #Semi-Supervised Object Detection
    if ssod_method is not None:
        if ssod_method == 'DenseTeacher':
            trainer = Trainer_DenseTeacher(cfg_dict, mode='train')
        elif ssod_method == 'ARSL':
            trainer = Trainer_ARSL(cfg_dict, mode='train')
        elif ssod_method == 'Semi_RTDETR':
            trainer = Trainer_Semi_RTDETR(cfg_dict, mode='train')
        else:
            raise ValueError(
                "Semi-Supervised Object Detection only no support this method.")
# cot 可能代表的是以下某种类型的训练策略：

# Co-Training（协同训练）：协同训练是一种半监督学习方法，其中两个模型（或两个学习器）共同训练，彼此提供标签和反馈，以相互改进。它通常用于处理标签稀缺的情况。具体来说，两个模型分别使用不同的视角或不同的数据子集进行训练，通过互相传递预测和标签来提高性能。

# Co-Tuning（协同调优）：协同调优是一种类似的策略，通常出现在迁移学习或多任务学习中。它可能意味着在多个任务之间共享部分网络或权重，通过训练过程中的相互调优来提升对不同任务的适应能力。例如，模型可能同时在基础任务（base task）和新任务（novel task）上进行训练，并通过优化两个任务的目标来调整模型的参数。

# Label-Cotuning（标签协同调优）：标签协同调优可能是一种专门用于少样本学习或迁移学习的技术，在这种技术中，模型学习如何基于已知类别的标签和新类别的标签之间的关系来优化预测。这与 Co-Tuning 有一定相似之处，重点在于如何通过标签之间的相互关系来提高模型对新类别的分类能力。
    elif cfg_dict.get('use_cot', False):
        trainer = TrainerCot(cfg_dict, mode='train')
    else:
        trainer = Trainer(cfg_dict, mode='train')

    # load weights
    if FLAGS.resume is not None:
        trainer.resume_weights(FLAGS.resume)
    elif 'pretrain_student_weights' in cfg_dict and 'pretrain_teacher_weights' in cfg_dict \
            and cfg_dict.pretrain_teacher_weights and cfg_dict.pretrain_student_weights:
        trainer.load_semi_weights(cfg_dict.pretrain_teacher_weights,
                                  cfg_dict.pretrain_student_weights)
    elif 'pretrain_weights' in cfg_dict and cfg_dict.pretrain_weights:
        trainer.load_weights(cfg_dict.pretrain_weights)

    # training
    trainer.train(FLAGS.eval)


def main():
    FLAGS = parse_args() #主观上只传入整体的配置文件
    cfg_dict = load_config(FLAGS.config)
    merge_args(cfg_dict, FLAGS) #融入手动传入的参数
    # merge_config(FLAGS.opt)

    # disable npu in config by default
    if 'use_npu' not in cfg_dict:
        cfg_dict.use_npu = False

    # disable xpu in config by default
    if 'use_xpu' not in cfg_dict:
        cfg_dict.use_xpu = False

    if 'use_gpu' not in cfg_dict:
        cfg_dict.use_gpu = False

    # disable mlu in config by default
    if 'use_mlu' not in cfg_dict:
        cfg_dict.use_mlu = False

    if cfg_dict.use_gpu:
        place = paddle.set_device('gpu')
    elif cfg_dict.use_npu:
        place = paddle.set_device('npu')
    elif cfg_dict.use_xpu:
        place = paddle.set_device('xpu')
    elif cfg_dict.use_mlu:
        place = paddle.set_device('mlu')
    else:
        place = paddle.set_device('cpu') #paddle内部函数

    if FLAGS.slim_config:
        cfg_dict = build_slim_model(cfg_dict, FLAGS.slim_config)

    # FIXME: Temporarily solve the priority problem of FLAGS.opt
    # merge_config(FLAGS.opt)
    check.check_config(cfg_dict)
    check.check_gpu(cfg_dict.use_gpu)
    check.check_npu(cfg_dict.use_npu)
    check.check_xpu(cfg_dict.use_xpu)
    check.check_mlu(cfg_dict.use_mlu)
    check.check_version()

    run(FLAGS, cfg_dict)


if __name__ == "__main__":
    main()
