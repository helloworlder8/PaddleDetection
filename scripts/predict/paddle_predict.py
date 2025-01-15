import os
import sys
import glob
import ast
# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import paddle
from ppdet.core.workspace import create, load_config, merge_config
from ppdet.engine import Trainer, Trainer_ARSL
import ppdet.utils.check as check
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.slim import build_slim_model

from ppdet.utils.logger import setup_logger
logger = setup_logger('train')

        # self.FLAGS = parse_args() #主观上只传入整体的配置文件
        # self.cfg_dict = load_config(self.FLAGS.config)
        # self.cfg_dict = merge_args(self.cfg_dict, self.FLAGS) #融入手动传入的参数
        # self.place = self._set_device()
        # self._build_trainer()
        


        
def get_test_images(infer_dir, infer_img, infer_list=None):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    if infer_list:
        assert os.path.isfile(
            infer_list), f"infer_list {infer_list} is not a valid file path."
        with open(infer_list, 'r') as f:
            lines = f.readlines()
        for line in lines:
            images.update([os.path.join(infer_dir, line.strip())])
    else:
        exts = ['jpg', 'jpeg', 'png', 'bmp']
        exts += [ext.upper() for ext in exts]
        for ext in exts:
            images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)
    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images



def parse_args():
    parser = ArgsParser()
    parser.add_argument("--weights", type=str, default=None, help="Combine method matching metric, choose in ['iou', 'ios'].")
    parser.add_argument("--infer_dir", type=str, default=None, help="Directory for images to perform inference on.")
    parser.add_argument("--infer_list", type=str, default=None, help="The file path containing path of image to be infered. Valid only when --infer_dir is given."
    )
    parser.add_argument("--infer_img", type=str, default=None, help="Image path, has higher priority over --infer_dir")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for storing the output visualization files.")
    parser.add_argument("--draw_threshold", type=float, default=0.5, help="Threshold to reserve the result for visualization.")
    parser.add_argument("--save_threshold", type=float, default=0.5, help="Threshold to reserve the result for saving.")
    parser.add_argument("--slim_config", default=None, type=str, help="Configuration file of slim method.")
    parser.add_argument("--use_vdl", type=bool, default=False, help="Whether to record the data to VisualDL.")
    parser.add_argument("--do_eval", type=ast.literal_eval, default=False, help="Whether to eval after infer.")
    parser.add_argument('--vdl_log_dir', type=str, default="vdl_log_dir/image", help='VisualDL logging directory for image.')
    parser.add_argument("--save_results", type=bool, default=False, help="Whether to save inference results to output_dir.")
    parser.add_argument("--slice_infer", action='store_true',  help="Whether to slice the image and merge the inference results for small object detection."
    )
    parser.add_argument('--slice_size',nargs='+', type=int, default=[640, 640], help="Height of the sliced image.")
    parser.add_argument( "--overlap_ratio", nargs='+', type=float, default=[0.25, 0.25], help="Overlap height ratio of the sliced image.")
    parser.add_argument( "--combine_method", type=str, default='nms', help="Combine method of the sliced images' detection results, choose in ['nms', 'nmm', 'concat']."
    )
    parser.add_argument("--match_threshold", type=float, default=0.6, help="Combine method matching threshold.")
    parser.add_argument("--match_metric", type=str, default='ios', help="Combine method matching metric, choose in ['iou', 'ios'].")
    parser.add_argument( "--visualize", type=ast.literal_eval, default=True, help="Whether to save visualize results to output_dir.")
    parser.add_argument( "--rtn_im_file", type=bool, default=False, help="Whether to return image file path in Dataloader.")
    args = parser.parse_args()
    return args
    
class PaddleDetectionPredictor:
    def __init__(self):
        """Initialize inference settings and load configuration."""
        self.FLAGS = parse_args()
        self.cfg_dict = load_config(self.FLAGS.config)
        merge_args(self.cfg_dict, self.FLAGS)
        self._set_device()

        self._build_Informer()



    def _set_device(self):
        """Set up the device based on configuration settings."""
        """Set the device based on configuration."""

        # Disable device flags (gpu, npu, xpu, mlu) in config by default
        for device in ['use_npu', 'use_xpu', 'use_gpu', 'use_mlu']:
            if device not in self.cfg_dict:
                setattr(self.cfg_dict, device, False)


        # Set device based on enabled flag priority (gpu > npu > xpu > mlu > cpu) 使用gpu是全局默认配置
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

    def _build_Informer(self):
        """Initialize inferer based on the configuration."""
        if self.FLAGS.rtn_im_file:
            self.cfg_dict['TestReader']['sample_transforms'][0]['Decode'][
                'rtn_im_file'] = self.FLAGS.rtn_im_file
        ssod_method = self.cfg_dict.get('ssod_method', None)
        if ssod_method == 'ARSL':
            self.trainer = Trainer_ARSL(self.cfg_dict, mode='test')
            self.trainer.load_weights(self.cfg_dict.weights, ARSL_eval=True)
        else:
            self.trainer = Trainer(self.cfg_dict, mode='test')
            self.trainer.load_weights(self.cfg_dict.weights)
        # get inference images
        if self.FLAGS.do_eval:
            dataset = create('TestDataset')()
            self.images = dataset.get_images()
        else:
            self.images = get_test_images(self.FLAGS.infer_dir, self.FLAGS.infer_img, self.FLAGS.infer_list)

    def predict(self):
        if self.FLAGS.slice_infer:
            self.trainer.slice_predict(
                self.images,
                slice_size=self.FLAGS.slice_size,
                overlap_ratio=self.FLAGS.overlap_ratio,
                combine_method=self.FLAGS.combine_method,
                match_threshold=self.FLAGS.match_threshold,
                match_metric=self.FLAGS.match_metric,
                draw_threshold=self.FLAGS.draw_threshold,
                output_dir=self.FLAGS.output_dir,
                save_results=self.FLAGS.save_results,
                visualize=self.FLAGS.visualize)
        else:
            self.trainer.predict(
                self.images,
                draw_threshold=self.FLAGS.draw_threshold, #0.5
                output_dir=self.FLAGS.output_dir, #out
                save_results=self.FLAGS.save_results, #False
                visualize=self.FLAGS.visualize, #true
                save_threshold=self.FLAGS.save_threshold, #0.5
                do_eval=self.FLAGS.do_eval) #要不要评估


