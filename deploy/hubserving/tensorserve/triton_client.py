import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.http import InferenceServerClient
from tritonclient.utils import triton_to_np_dtype



# model_name = 'onnx'
# URL='20.230.128.239:8001'
# inputs 
class triton_repo:
    def __init__(
        self,
        URL = '',
        model_name = '',
        model_version = '1'
    ):

        self.model_version = model_version
        self.URL = URL
        self.model_name = model_name
        self.triton_client = grpcclient.InferenceServerClient(url=URL)
        self.model_metadata = self.triton_client.get_model_metadata(model_name=model_name, 
                                                    model_version=self.model_version)
        self.m_inputs = self.model_metadata.inputs #['inputs']
        
    # GRPC Calls
    def get_feature_sorted(self,
        key, 
        feed_dict, 
        f_cnf, 
        batch=1
    ):
        array  = feed_dict[key] if isinstance(feed_dict[key], np.ndarray) else feed_dict[key].numpy()
        # array = feed_dict[key].numpy()
        # array = np.expand_dims(array, axis=0)
        array = array.astype(triton_to_np_dtype(f_cnf.datatype)) #['datatype']))
        feature = grpcclient.InferInput( key, list(array.shape), f_cnf.datatype)#['datatype'])
        feature.set_data_from_numpy(array)
        return feature

        

    def triton_inference(self, inputs):

        # import pickle
        # with open('traj.pkl', 'rb') as f:
        #     initial_frame = pickle.load(f)

        feed_dict = inputs.copy()
        inputs1 = []
        for f_cnf in self.m_inputs:
            inputs1.append(self.get_feature_sorted(f_cnf.name, feed_dict, f_cnf))

        outputs = []
        for f in self.model_metadata.outputs: 
            outputs.append(grpcclient.InferRequestedOutput(f.name))
            
        # Test with outputs
        results = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs1,
            outputs=outputs,
            headers={'test': '1'})
        # response = results.get_response()
        return results