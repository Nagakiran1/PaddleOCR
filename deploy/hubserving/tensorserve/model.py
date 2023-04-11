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

import os
import sys
sys.path.insert(0, ".")
import copy

import time
import paddlehub
# from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, runnable, serving
import cv2
import paddlehub as hub
import numpy as np

from utility import base64_to_cv2
from predict_layout import LayoutPredictor as _LayoutPredictor
# from ppstructure.utility import parse_args

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

from utility import draw_ocr_box_txt, str2bool, init_args as infer_args


def init_args():
    parser = infer_args()

    # params for output
    parser.add_argument("--output", type=str, default='./output')
    # params for table structure
    parser.add_argument("--table_max_len", type=int, default=488)
    parser.add_argument("--table_algorithm", type=str, default='TableAttn')
    parser.add_argument("--table_model_dir", type=str)
    parser.add_argument(
        "--merge_no_span_structure", type=str2bool, default=True)
    parser.add_argument(
        "--table_char_dict_path",
        type=str,
        default="../ppocr/utils/dict/table_structure_dict_ch.txt")
    # params for layout
    parser.add_argument("--layout_model_dir", type=str)
    parser.add_argument(
        "--layout_dict_path",
        type=str,
        default="../ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt")
    parser.add_argument(
        "--layout_score_threshold",
        type=float,
        default=0.5,
        help="Threshold of score.")
    parser.add_argument(
        "--layout_nms_threshold",
        type=float,
        default=0.5,
        help="Threshold of nms.")
    # params for kie
    parser.add_argument("--kie_algorithm", type=str, default='LayoutXLM')
    parser.add_argument("--ser_model_dir", type=str)
    parser.add_argument("--re_model_dir", type=str)
    parser.add_argument("--use_visual_backbone", type=str2bool, default=True)
    parser.add_argument(
        "--ser_dict_path",
        type=str,
        default="../train_data/XFUND/class_list_xfun.txt")
    # need to be None or tb-yx
    parser.add_argument("--ocr_order_method", type=str, default=None)
    # params for inference
    parser.add_argument(
        "--mode",
        type=str,
        choices=['structure', 'kie'],
        default='structure',
        help='structure and kie is supported')
    parser.add_argument(
        "--image_orientation",
        type=bool,
        default=False,
        help='Whether to enable image orientation recognition')
    parser.add_argument(
        "--layout",
        type=str2bool,
        default=True,
        help='Whether to enable layout analysis')
    parser.add_argument(
        "--table",
        type=str2bool,
        default=True,
        help='In the forward, whether the table area uses table recognition')
    parser.add_argument(
        "--ocr",
        type=str2bool,
        default=True,
        help='In the forward, whether the non-table area is recognition by ocr')
    # param for recovery
    parser.add_argument(
        "--recovery",
        type=str2bool,
        default=False,
        help='Whether to enable layout of recovery')
    parser.add_argument(
        "--use_pdf2docx_api",
        type=str2bool,
        default=False,
        help='Whether to use pdf2docx api')

    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()

class Config(object):
    pass


def read_params():
    cfg = Config()

    # params for layout analysis
    cfg.layout_model_dir = './picodet_lcnet_x1_0_layout_infer/'
    cfg.layout_dict_path = 'layout_publaynet_dict.txt'
    cfg.layout_score_threshold = 0.5
    cfg.layout_nms_threshold = 0.5
    return cfg


@moduleinfo(
    name="structure_layout",
    version="1.0.0",
    summary="PP-Structure layout service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/structure_layout")
class LayoutPredictor(hub.Module):
    def _initialize(self, use_gpu=False, enable_mkldnn=False):
        """
        initialize with the necessary elements
        """
        cfg = self.merge_configs()
        cfg.use_gpu = use_gpu
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
                print("use gpu: ", use_gpu)
                print("CUDA_VISIBLE_DEVICES: ", _places)
                cfg.gpu_mem = 8000
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
        cfg.ir_optim = True
        cfg.enable_mkldnn = enable_mkldnn

        self.layout_predictor = _LayoutPredictor(cfg)

    def merge_configs(self):
        # deafult cfg
        backup_argv = copy.deepcopy(sys.argv)
        sys.argv = sys.argv[:1]
        cfg = parse_args()

        update_cfg_map = vars(read_params())

        for key in update_cfg_map:
            cfg.__setattr__(key, update_cfg_map[key])

        sys.argv = copy.deepcopy(backup_argv)
        return cfg

    def read_images(self, paths=[]):
        images = []
        for img_path in paths:
            assert os.path.isfile(
                img_path), "The {} isn't a valid file.".format(img_path)
            img = cv2.imread(img_path)
            if img is None:
                # logger.info("error in loading image:{}".format(img_path))
                continue
            images.append(img)
        return images

    def predict(self, predicted_data=[], paths=[]):
        """
        Get the chinese texts in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
        Returns:
            res (list): The layout results of images.
        """

        # if images != [] and isinstance(images, list) and paths == []:
        #     predicted_data = images
        # elif images == [] and isinstance(paths, list) and paths != []:
        #     predicted_data = self.read_images(paths)
        # else:
        #     raise TypeError("The input data is inconsistent with expectations.")

        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."

        all_results = []
        for img in predicted_data:
            if img is None:
                # logger.info("error in loading image")
                all_results.append([])
                continue
            starttime = time.time()
            res, _ = self.layout_predictor(img)
            elapse = time.time() - starttime
            # logger.info("Predict time: {}".format(elapse))

            for item in res:
                item['bbox'] = item['bbox'].tolist()
            all_results.append({'layout': res})
        return all_results

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.predict(images_decode, **kwargs)
        return results


# if __name__ == '__main__':
#     layout = LayoutPredictor()
#     layout._initialize()
    
#     image_path = ['./1.png','./1.png']
#     input_images = layout.read_images(paths=image_path)
#     res = layout.predict(predicted_data=input_images)
    
#     label_map = {'text':0 ,'title': 1 ,  'list':2, 'table':3, 'figure':4}
#     im_bbox = []
#     im_ids = []
#     im_class = []
#     for im_id, im_res in enumerate(res):
#         for bound_id in range(len(im_res['layout'])):
#             im_bbox.append(im_res['layout'][bound_id]['bbox'])
#             im_class.append(label_map[im_res['layout'][bound_id]['label']])
#             im_ids.append(im_id)
#     im_bbox = np.vstack(im_bbox)
#     im_ids = np.vstack(im_ids)
#     im_class = np.vstack(im_class)
#     from IPython.core.debugger import set_trace
#     set_trace()
            
        
    print(res)
    















# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
from torch import nn
import torch
from torch.utils.dlpack import from_dlpack

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class AddSubNet(nn.Module):
    """
    Simple AddSub network in PyTorch. This network outputs the sum and
    subtraction of the inputs.
    """

    def __init__(self):
        super(AddSubNet, self).__init__()

    def forward(self, input0, input1):
        return (input0 + input1), (input0 - input1)


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "pred_boxes")

        # Get OUTPUT1 configuration
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "pred_classes")

        # Get OUTPUT2 configuration
        output2_config = pb_utils.get_output_config_by_name(
            model_config, "pred_id")
        
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])
        self.output2_dtype = pb_utils.triton_string_to_numpy(
                    output2_config['data_type'])

        self.label_map = {'Text':0 ,'Title': 1 ,  'List':2, 'Table':3, 'Figure':4}
        # Instantiate the PyTorch model
        # self.add_sub_model = AddSubNet()
        self.layout = LayoutPredictor()
        self.layout._initialize()
        
    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype
        output2_dtype = self.output2_dtype


        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "image")
            
            # out_0, out_1 = self.add_sub_model(in_0.as_numpy(), in_1.as_numpy())
            in_0 = in_0.as_numpy()
            inputs = [in_0[i] for i in range(len(in_0))] # [{'image':in_0.as_numpy()}]

            res = self.layout.predict(predicted_data=inputs)
    
            im_bbox = []
            im_ids = []
            im_class = []
            for im_id, im_res in enumerate(res):
                for bound_id in range(len(im_res['layout'])):
                    im_bbox.append(im_res['layout'][bound_id]['bbox'])
                    im_class.append(self.label_map[im_res['layout'][bound_id]['label']])
                    im_ids.append(im_id)
            out_0 = np.vstack(im_bbox)
            out_1 = np.vstack(im_ids)
            out_2 = np.vstack(im_class)
            
            
            # val = self.detectron(inputs)
            # out_0 = val[0]['instances'].pred_boxes.tensor.cpu().detach().numpy()
            # out_1 = val[0]['instances'].pred_classes.cpu().detach().numpy()
            # out_2 = val[0]['instances'].scores.cpu().detach().numpy()
            
            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("pred_boxes",
                                           out_0.astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("pred_classes",
                                           out_1.astype(output1_dtype))
            out_tensor_2 = pb_utils.Tensor("pred_id",
                                           out_2.astype(output2_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1, out_tensor_2])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')



