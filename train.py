
from torch.utils.data import DataLoader
from utils.train_utils import *
import argparse
import pycrayon
import numpy as np
from utils.dataset_utils import *
import torch.nn.utils as utils
import os
from datasets.MPII import MPII
from utils.evaluation import accuracy
from models.RecurrentStackedHourglass import PretrainRecurrentStackedHourglass
from utils.augmentation import ImageTransformer
import torch
from tqdm import tqdm
from models.MSESequenceLoss import MSESequenceLoss


def train(model, loader, criterion, optimizer, scheduler, device, summary=None):
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
            utils.clip_grad_value_(model.parameters(), 100)
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
    valid_transformer = ImageTransformer(p_scale=0.0, p_flip=0.0, p_rotate=0.0)

    train_dataset = MPII(root=mpii_root, transformer=train_transformer, output_size=args.resolution, train=True)
    valid_dataset = MPII(root=mpii_root, transformer=valid_transformer, output_size=args.resolution, train=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, **loader_args)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, **loader_args)

    model = PretrainRecurrentStackedHourglass(3, 64, train_dataset.n_joints + 1, device, T=args.t, depth=args.depth)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_interval, gamma=args.lr_drop_factor)
    criterion = MSESequenceLoss().to(device)

    summary = None
    if args.use_tensorboard:
        cc = pycrayon.CrayonClient(hostname='localhost')
        summary = cc.create_experiment('Recurrent Stacked Hourglass Training')

    best_acc = 0.
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(snapshot_name)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['accuracy']
        if args.use_tensorboard:
            summary.add_scalar_value('Epoch Valid Accuracy', checkpoint['accuracy'])
            summary.add_scalar_value('Epoch Valid Loss', checkpoint['loss'])

    for epoch in range(start_epoch, args.max_epochs):
        print('\n[epoch ' + str(epoch) + ']')
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, device, summary)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)

        if args.use_tensorboard:
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

    parser.add_argument('--t', default=10, type=int)
    parser.add_argument('--depth', default=4, type=int)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_drop_interval', default=100000, type=int)
    parser.add_argument('--lr_drop_factor', default=0.333, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--decay', default=5e-3, type=float)
    parser.add_argument('--max_epochs', default=5000, type=int)
    parser.add_argument('--resolution', default=256, type=int)

    parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--device', default='cpu', type=str, choices=['cpu', '0', '1'])
    main(parser.parse_args())
