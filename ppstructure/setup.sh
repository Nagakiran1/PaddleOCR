# python3 -m pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# pip install paddleclas

# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/deploy/paddle2onnx/readme.md


# ### CUDANN8 DEVEL ###
# # https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.3.1/ubuntu2004/devel/cudnn8/Dockerfile

# export NV_CUDNN_VERSION=8.2.0.53
# export NV_CUDNN_PACKAGE_NAME="libcudnn8"

# export NV_CUDNN_PACKAGE="libcudnn8=$NV_CUDNN_VERSION-1+cuda11.3"
# export NV_CUDNN_PACKAGE_DEV="libcudnn8-dev=$NV_CUDNN_VERSION-1+cuda11.3"


# apt-get update
# apt-get install -y --no-install-recommends 
# apt-get install ${NV_CUDNN_PACKAGE}
# apt-get install ${NV_CUDNN_PACKAGE_DEV} 
# apt-get install && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} 

# # Cleanup
# apt-get clean 
# rm -rf /root/.cache/* 
# rm -rf /tmp/* 
# apt-get install 
# rm -rf /var/lib/apt/lists/*



# # model_conversion -  https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/deploy/paddle2onnx/readme.md#2-model-conversion

# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/inference_ppocr_en.md

pip install paddle2onnx
pip install onnxruntime-gpu
mkdir inference && cd inference
# Download the PP-StructureV2 layout analysis model and unzip it
wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_layout_infer.tar && tar xf picodet_lcnet_x1_0_layout_infer.tar
# Download the PP-OCRv3 text detection model and unzip it
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar && tar xf ch_PP-OCRv3_det_infer.tar
# Download the PP-OCRv3 text recognition model and unzip it
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar && tar xf ch_PP-OCRv3_rec_infer.tar
# Download the PP-StructureV2 form recognition model and unzip it
wget https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar && tar xf ch_ppstructure_mobile_v2.0_SLANet_infer.tar
cd ..


paddle2onnx --model_dir ./inference/ch_PP-OCRv3_det_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/ch_PP-OCRv3_det_infer/model.onnx \
--opset_version 11 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True

paddle2onnx --model_dir ./inference/ch_PP-OCRv3_rec_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/ch_PP-OCRv3_rec_infer/model.onnx \
--opset_version 10 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True

paddle2onnx --model_dir ./inference/ch_ppstructure_mobile_v2.0_SLANet_infer \
--model_filename ch_ppstructure_mobile_v2.0_SLANet_infer/inference.pdmodel \
--params_filename ch_ppstructure_mobile_v2.0_SLANet_infer/inference.pdiparams \
--save_file ./inference/ch_ppstructure_mobile_v2.0_SLANet_infer/model.onnx \
--opset_version 10 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True

paddle2onnx --model_dir ./inference/picodet_lcnet_x1_0_layout_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/picodet_lcnet_x1_0_layout_infer/model.onnx \
--opset_version 10 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True