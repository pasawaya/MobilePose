
import torch.onnx
import onnx_coreml


def save_coreml(model, dummy_input, model_name):
    onnx_model_name = 'model.onnx'
    torch.onnx.export(model, dummy_input, onnx_model_name)
    mlmodel = onnx_coreml.convert(onnx_model_name,
                                  mode='regressor',
                                  image_input_names='0',
                                  image_output_names='309',
                                  predicted_feature_name='keypoints')
    mlmodel.save(model_name)