import onnx
import onnxruntime as ort
import numpy as np

def test(observation_size):
    onnx_path = "my_ppo_model.onnx"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    observation = np.zeros((1, *observation_size)).astype(np.float32)
    ort_sess = ort.InferenceSession(onnx_path)
    action, value = ort_sess.run(None, {"input": observation})