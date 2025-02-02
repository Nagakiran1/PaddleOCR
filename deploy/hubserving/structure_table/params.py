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

from deploy.hubserving.ocr_system.params import read_params as pp_ocr_read_params


def read_params():
    cfg = pp_ocr_read_params()

    # params for table structure model
    cfg.table_max_len = 488
    cfg.table_model_dir = '/home/ubuntu/codespaces/temp/PaddleOCR/ppstructure/inference/ch_ppstructure_mobile_v2.0_SLANet_infer'
    cfg.det_model_dir = '/home/ubuntu/codespaces/temp/PaddleOCR/ppstructure/inference/ch_PP-OCRv3_det_infer'
    cfg.table_char_dict_path = './ppocr/utils/dict/table_structure_dict.txt'
    cfg.show_log = False
    return cfg
