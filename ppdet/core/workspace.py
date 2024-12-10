# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function
from __future__ import division

import importlib
import os
import sys

import yaml
import collections

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

from .config.schema import SchemaDict, SharedConfig, extract_schema
from .config.yaml_helpers import serializable

__all__ = [
    'global_config',
    'load_config',
    'merge_config',
    'get_registered_modules',
    'create',
    'register',
    'serializable',
    'dump_value',
]


def dump_value(value):
    # XXX this is hackish, but collections.abc is not available in python 2
    if hasattr(value, '__dict__') or isinstance(value, (dict, tuple, list)):
        value = yaml.dump(value, default_flow_style=True)
        value = value.replace('\n', '')
        value = value.replace('...', '')
        return "'{}'".format(value)
    else:
        # primitive types
        return str(value)


class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs): #初始化键值对
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key): #得到属性
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))

    def __setattr__(self, key, value): #设置属性
        self[key] = value

    def copy(self): #拷贝成一个新类
        new_dict = AttrDict()
        for k, v in self.items():
            new_dict.update({k: v})
        return new_dict

#在这个文件内的全局配置
global_config = AttrDict() #322个全局变量

BASE_KEY = '_BASE_'


# parse and load _BASE_ recursively
def load_config_with_base(cfg_yaml): #递归加载文件
    with open(cfg_yaml) as f:
        part_cfg_dict = yaml.load(f, Loader=yaml.Loader)

    # NOTE: 外部比_BASE_有更高的优先级   
    if BASE_KEY in part_cfg_dict:
        cfg_dict = AttrDict()
        base_cfg_ymls = list(part_cfg_dict[BASE_KEY])
        for base_cfg_yml in base_cfg_ymls:
            if base_cfg_yml.startswith("~"):
                base_cfg_yml = os.path.expanduser(base_cfg_yml)
            if not base_cfg_yml.startswith('/'): #不是绝对路径  也就是说是相对路径
                base_cfg_yml = os.path.join(os.path.dirname(cfg_yaml), base_cfg_yml)

            with open(base_cfg_yml) as f:
                base_cfg_dict = load_config_with_base(base_cfg_yml) #递归
                cfg_dict = merge_config(base_cfg_dict, cfg_dict)
        # 有基础但是基础都处理完了
        del part_cfg_dict[BASE_KEY]
        return merge_config(part_cfg_dict, cfg_dict) #346个全局变量  这里是真正的返回

    return part_cfg_dict #最后也会从这里出去


def load_config(cfg_yaml): #整体配置文件

    # load base_cfg_dict from file and merge into global base_cfg_dict
    cfg_dict = load_config_with_base(cfg_yaml)
    cfg_dict['filename'] = os.path.splitext(os.path.split(cfg_yaml)[-1])[0] #全局配置名
    merge_config(cfg_dict) #加一个文件名

    return global_config

# base_cfg_dict
def dict_merge(base_cfg_dict, cfg_dict):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``base_cfg_dict`` is merged into
    ``dct``.

    Args:
        dct: dict onto which the merge is executed
        base_cfg_dict: dct merged into dct

    Returns: dct
    """
    for k, v in base_cfg_dict.items():
        if (k in cfg_dict and isinstance(cfg_dict[k], dict) and
                isinstance(base_cfg_dict[k], collectionsAbc.Mapping)):
            dict_merge(base_cfg_dict[k], cfg_dict[k]) #再次回调
        else:
            cfg_dict[k] = base_cfg_dict[k]
    return cfg_dict

# base_cfg_dict, cfg_dict)
def merge_config(base_cfg_dict, cfg_dict=None):
    """
    Merge base_cfg_dict into global base_cfg_dict or cfg_dict.

    Args:
        base_cfg_dict (dict): Config to be merged.

    Returns: global base_cfg_dict
    """
    global global_config
    cfg_dict = cfg_dict or global_config
    return dict_merge(base_cfg_dict, cfg_dict)


def get_registered_modules():
    return {k: v for k, v in global_config.items() if isinstance(v, SchemaDict)}


def make_partial(cls):
    op_module = importlib.import_module(cls.__op__.__module__)
    op = getattr(op_module, cls.__op__.__name__)
    cls.__category__ = getattr(cls, '__category__', None) or 'op'

    def partial_apply(self, *args, **kwargs):
        kwargs_ = self.__dict__.copy()
        kwargs_.update(kwargs)
        return op(*args, **kwargs_)

    if getattr(cls, '__append_doc__', True):  # XXX should default to True?
        if sys.version_info[0] > 2:
            cls.__doc__ = "Wrapper for `{}` OP".format(op.__name__)
            cls.__init__.__doc__ = op.__doc__
            cls.__call__ = partial_apply
            cls.__call__.__doc__ = op.__doc__
        else:
            # XXX work around for python 2
            partial_apply.__doc__ = op.__doc__
            cls.__call__ = partial_apply
    return cls

# 全局的注册机
def register(cls):
    """
    Register a given module class.

    Args:
        cls (type): Module class to be registered.

    Returns: cls
    """
    if cls.__name__ in global_config:
        raise ValueError("Module class already registered: {}".format(
            cls.__name__))
    if hasattr(cls, '__op__'):
        cls = make_partial(cls)
    global_config[cls.__name__] = extract_schema(cls)
    return cls


def create(cls_strcls, **kwargs): # 'TrainDataset' 首先开始的是创建训练数据集
    """
    Create an instance of given module class.

    Args:
        cls_strcls (type or str): Class of which to create instance.

    Returns: instance of type `cls_strcls`
    """
    assert type(cls_strcls) in [type, str], "should be a class or name of a class"
    name = type(cls_strcls) == str and cls_strcls or cls_strcls.__name__
    if name in global_config:
        if isinstance(global_config[name], SchemaDict): #简单理解为一个字典就行
            pass
        elif hasattr(global_config[name], "__dict__"):
            # support instance return directly
            return global_config[name]
        else:
            raise ValueError("The module {} is not registered".format(name))
    else:
        raise ValueError("The module {} is not registered".format(name))

    base_cfg_dict = global_config[name]
    cls = getattr(base_cfg_dict.pymodule, name) #类名
    cls_kwargs = {}
    cls_kwargs.update(global_config[name]) #类参数

    # parse `shared` annoation of registered modules
    if getattr(base_cfg_dict, 'shared', None):
        for k in base_cfg_dict.shared: #共享参数赋值
            target_value = base_cfg_dict[k]
            shared_conf = base_cfg_dict.schema[k].default
            assert isinstance(shared_conf, SharedConfig)
            if target_value is not None and not isinstance(target_value, SharedConfig):
                continue  # value is given for the module
            elif shared_conf.key in global_config:
                # `key` is present in base_cfg_dict
                cls_kwargs[k] = global_config[shared_conf.key]
            else:
                cls_kwargs[k] = shared_conf.default_value

    # parse `inject` annoation of registered modules
    if getattr(cls, 'from_config', None):
        cls_kwargs.update(cls.from_config(base_cfg_dict, **kwargs))

    if getattr(base_cfg_dict, 'inject', None):
        for k in base_cfg_dict.inject:
            target_value = base_cfg_dict[k]
            # optional dependency
            if target_value is None:
                continue

            if isinstance(target_value, dict) or hasattr(target_value, '__dict__'):
                if 'name' not in target_value.keys():
                    continue
                inject_name = str(target_value['name'])
                if inject_name not in global_config:
                    raise ValueError(
                        "Missing injection name {} and check it's name in cfg file".
                        format(k))
                target = global_config[inject_name]
                for i, v in target_value.items():
                    if i == 'name':
                        continue
                    target[i] = v
                if isinstance(target, SchemaDict):
                    cls_kwargs[k] = create(inject_name) #又是一层递归
            elif isinstance(target_value, str):
                if target_value not in global_config:
                    raise ValueError("Missing injection base_cfg_dict:", target_value)
                target = global_config[target_value]
                if isinstance(target, SchemaDict):
                    cls_kwargs[k] = create(target_value)
                elif hasattr(target, '__dict__'):  # serialized object
                    cls_kwargs[k] = target
            else:
                raise ValueError("Unsupported injection type:", target_value)
    # prevent modification of global base_cfg_dict values of reference types
    # (e.g., list, dict) from within the created module instances
    #kwargs = copy.deepcopy(kwargs)
    return cls(**cls_kwargs) #类地址 参数
