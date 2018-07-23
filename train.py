
from torch.utils.data import DataLoader
from utils.train_utils import *
import argparse
import pycrayon
from utils.dataset_utils import *
import torch.nn.utils as utils
import os
from datasets.MPII import MPII
from utils.evaluation import accuracy
from models.RecurrentStackedHourglass import PretrainRecurrentStackedHourglass
from models.RSHDeploy import RecurrentStackedHourglass
from utils.augmentation import ImageTransformer
import torch
import torch.onnx
from tqdm import tqdm
from models.MSESequenceLoss import MSESequenceLoss
from models.modules.ConvolutionalBlock import ConvolutionalBlock
from onnx_coreml import convert
from models.modules.ResidualBlock import ResidualBlock
from models.modules.InvertedResidualBlock import InvertedResidualBlock
from models.modules.ConvolutionalBlock import ConvolutionalBlock


def train(model, loader, criterion, optimizer, scheduler, device, clip=None, summary=None):
    loss_avg = RunningAverage()
    acc_avg = RunningAverage()

    model.train()

    with tqdm(total=len(loader)) as t:
        for i, (frames, label_map, centers, _) in enumerate(loader):
            frames, label_map, centers = frames.to(device), label_map.to(device), centers.to(device)

            outputs = model(frames, centers)
            loss = criterion(outputs, label_map)
            acc = accuracy(outputs, label_map)

            optimizer.zero_grad()
            loss.backward()
            if clip is not None:
                utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()

            loss_avg.update(loss.item())
            acc_avg.update(acc)

            if summary is not None:
                summary.add_scalar_value('Train Accuracy', acc)
                summary.add_scalar_value('Train Loss', loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss.item()), acc='{:05.3f}%'.format(acc * 100),
                          loss_avg='{:05.3f}'.format(loss_avg()), acc_avg='{:05.3f}%'.format(acc_avg() * 100))
            t.update()

        return loss_avg(), acc_avg()


def validate(model, loader, criterion, device):
    loss_avg = RunningAverage()
    acc_avg = RunningAverage()

    model.eval()
    for i, (frames, label_map, centers, _) in enumerate(loader):
        frames, label_map, centers = frames.to(device), label_map.to(device), centers.to(device)

        outputs = model(frames, centers)
        loss = criterion(outputs, label_map)
        acc = accuracy(outputs, label_map)

        loss_avg.update(loss.item())
        acc_avg.update(acc)

    return loss_avg(), acc_avg()


def main(args):
    snapshot_name = 'checkpoint.pth.tar'

    device_name = 'cpu' if args.device == 'cpu' else 'cuda:' + args.device
    device = torch.device(device_name)
    loader_args = {'num_workers': 1, 'pin_memory': True} if 'cuda' in device_name else {}

    mpii_root = 'data/MPII'
    mean_name = 'means.npy'
    mean_path = os.path.join(mpii_root, mean_name)
    if not os.path.isfile(mean_path):
        mpii = MPII(root=mpii_root, transformer=None, output_size=args.resolution, train=True)
        mean, std = compute_mean(mpii)
        np.save(mean_path, np.array([mean, std]))

    mean, std = np.load(os.path.join(mpii_root, 'means.npy'))
    train_transformer = ImageTransformer(mean=mean, std=std)
    valid_transformer = ImageTransformer(p_scale=0.0, p_flip=0.0, p_rotate=0.0, mean=mean, std=std)

    train_dataset = MPII(root=mpii_root, transformer=train_transformer, output_size=args.resolution, train=True,
                         subset_size=args.subset_size)
    valid_dataset = MPII(root=mpii_root, transformer=valid_transformer, output_size=args.resolution, train=False,
                         subset_size=args.subset_size)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, **loader_args)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, **loader_args)

    if args.block == 'residual':
        block = ResidualBlock
    elif args.block == 'inverted':
        block = InvertedResidualBlock
    else:
        block = ConvolutionalBlock
    model = PretrainRecurrentStackedHourglass(3, 64, train_dataset.n_joints + 1, device, block, T=args.t, depth=args.depth)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_interval, gamma=args.lr_drop_factor)
    criterion = MSESequenceLoss().to(device)

    summary = None
    if args.host is not None:
        cc = pycrayon.CrayonClient(hostname=args.host)
        if args.experiment in cc.get_experiment_names():
            summary = cc.open_experiment(args.experiment)
        else:
            summary = cc.create_experiment(args.experiment)

    best_acc = 0.
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(snapshot_name)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['accuracy']
        if args.host is not None:
            summary.add_scalar_value('Epoch Valid Accuracy', checkpoint['accuracy'])
            summary.add_scalar_value('Epoch Valid Loss', checkpoint['loss'])

    if args.coreml_name is not None:
        onnx_model_name = 'rsh.onnx'
        deploy_model = RecurrentStackedHourglass(3, 64, train_dataset.n_joints + 1, device, T=1, depth=2, block=ConvolutionalBlock)

        dummy_input = torch.FloatTensor(1, 3, 256, 256)
        torch.onnx.export(deploy_model, dummy_input, onnx_model_name)
        mlmodel = convert(onnx_model_name,
                          mode='regressor',
                          image_input_names='0',
                          image_output_names='309',
                          predicted_feature_name='keypoints')
        mlmodel.save(args.coreml_name)

    for epoch in range(start_epoch, args.max_epochs):
        print('\n[epoch ' + str(epoch) + ']')
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, device, args.clip, summary)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)

        if args.host is not None:
            summary.add_scalar_value('Epoch Train Loss', train_loss)
            summary.add_scalar_value('Epoch Valid Loss', valid_loss)
            summary.add_scalar_value('Epoch Train Accuracy', train_acc)
            summary.add_scalar_value('Epoch Valid Accuracy', valid_acc)
            summary.add_scalar_value('Learning Rate', scheduler.get_lr()[0])

        if valid_acc >= best_acc:
            best_acc = valid_acc
            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'accuracy': valid_acc,
                        'loss': valid_loss},
                       snapshot_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Architecture
    parser.add_argument('--t', default=10, type=int)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--block', default='residual', type=str, choices=['residual', 'inverted', 'conv'])

    # Training
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_drop_interval', default=100000, type=int)
    parser.add_argument('--lr_drop_factor', default=0.333, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--decay', default=0, type=float)
    parser.add_argument('--max_epochs', default=5000, type=int)
    parser.add_argument('--resolution', default=256, type=int)
    parser.add_argument('--subset_size', default=None, type=int)
    parser.add_argument('--clip', default=None, type=int)

    # Tensorboard
    parser.add_argument('--experiment', default='Recurrent Stacked Hourglass Training', type=str)
    parser.add_argument('--host', default=None, type=str)

    # Other
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--device', default='cpu', type=str, choices=['cpu', '0', '1'])
    parser.add_argument('--coreml_name', default=None, type=str)

    main(parser.parse_args())
