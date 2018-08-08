
# import torch.onnx
import onnx_coreml
import configargparse
import argparse
import os
import torch
from models.DeployPoseMachine import LPM


def save_coreml(model, dummy_input, onnx_model_name, mlmodel_name):
    # torch.onnx.export(model, dummy_input, onnx_model_name)
    mlmodel = onnx_coreml.convert(onnx_model_name,
                                  mode='regressor',
                                  image_input_names='0',
                                  image_output_names='309',
                                  predicted_feature_name='keypoints')
    mlmodel.save(mlmodel_name)


def main(args):
    device_name = 'cpu' if args.gpu is None else 'cuda:' + str(args.gpu)
    device = torch.device(device_name)

    n_joints = 14
    model = LPM(3, 32, n_joints + 1, device, T=args.t)

    if args.checkpoint_name is not None:
        print('Loading checkpoint...')
        path = os.path.join(args.model_dir, args.checkpoint_name)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])

    model = model.to(device)

    print('Exporting...')
    dummy_images = torch.zeros((1, args.t, 3, args.resolution, args.resolution)).to(device)
    dummy_centers = torch.zeros((1, 1, args.resolution, args.resolution)).to(device)
    save_coreml(model, (dummy_images, dummy_centers), args.onnx_name, args.core_ml_name)
    print('Done!')


if __name__ == '__main__':
    parser = configargparse.ArgParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add('--t', default=10, type=int, help='length of input sequences (i.e. # frames)')
    parser.add('--resolution', default=256, type=int, help='model input image resolution')
    parser.add('--checkpoint_name', default=None, type=str, help='checkpoint file name in experiments/ to resume from')
    parser.add('--model_dir', default='experiments', type=str, help='directory to store/load checkpoints from')
    parser.add('--onnx_name', default='model.onnx', type=str, help='name for onnx file')
    parser.add('--core_ml_name', default='model.mlmodel', type=str, help='name for mlmodel file')
    parser.add('--gpu', default=None, type=int, help='gpu id')

    main(parser.parse_args())
