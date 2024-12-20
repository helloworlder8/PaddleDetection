#!/bin/bash
code ../PaddleDetection -r ppdet/modeling/heads/centernet_head.py \
ppdet/modeling/reid/fairmot_embedding_head.py \
ppdet/data/reader.py
# def parse_model(model_dict, ch, verbose=True):