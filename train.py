
import argparse
import pycrayon
import time
import torch.nn.utils as utils
from tqdm import tqdm
from datasets.MPII import MPII

from torch.utils.data import DataLoader

from utils.train_utils import *
from utils.augmentation import ImageTransformer
from utils.dataset_utils import *
from utils.evaluation import accuracy

from models.RecurrentStackedHourglass import PretrainRecurrentStackedHourglass
from models.LSTMPoseMachine import PretrainLPM
from models.LSTMPoseCoordinatesMachine import PretrainCoordinateLPM
from models.modules.ResidualBlock import ResidualBlock
from models.modules.InvertedResidualBlock import InvertedResidualBlock
from models.modules.ConvolutionalBlock import ConvolutionalBlock
from models.losses.MSESequenceLoss import MSESequenceLoss
from models.losses.CoordinateLoss import CoordinateLoss


def train(model, loader, criterion, optimizer, device, scheduler=None, clip=None, summary=None):
    loss_avg = RunningAverage()
    acc_avg = RunningAverage()

    model.train()

    with tqdm(total=len(loader)) as t:
        for i, (frames, label_map, centers, meta) in enumerate(loader):
            frames, label_map, centers = frames.to(device), label_map.to(device), centers.to(device)
            outputs = model(frames, centers)
            if isinstance(criterion, CoordinateLoss):
                loss = criterion(*outputs, meta)
                acc = accuracy(outputs[0], label_map)
            else:
                loss = criterion(outputs, label_map)
                acc = accuracy(outputs, label_map)

            optimizer.zero_grad()
            loss.backward()
            if clip is not None:
                utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if scheduler is not None:
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
    model_dir = 'experiments'
    start_time_prefix = str(int(time.time()))[-4:] + "_"
    print('\nCheckpoint prefix will be ' + start_time_prefix)

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

    stride = 4 if args.model == 'hourglass' else 8
    offset = 0 if args.model == 'hourglass' else -1
    include_background = args.model != 'coord_lpm'
    train_dataset = MPII(root=mpii_root, transformer=train_transformer, output_size=args.resolution, train=True,
                         subset_size=args.subset_size, sigma=7, stride=stride, offset=offset,
                         include_background=include_background)
    valid_dataset = MPII(root=mpii_root, transformer=valid_transformer, output_size=args.resolution, train=False,
                         subset_size=args.subset_size, sigma=7, stride=stride, offset=offset,
                         include_background=include_background)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, **loader_args)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, **loader_args)

    if args.block == 'residual':
        block = ResidualBlock
    elif args.block == 'inverted':
        block = InvertedResidualBlock
    else:
        block = ConvolutionalBlock

    criterion = MSESequenceLoss()
    if args.model == 'hourglass':
        model = PretrainRecurrentStackedHourglass(3, 64, train_dataset.n_joints + 1, device, block, T=args.t, depth=args.depth)
    elif args.model == 'lpm':
        model = PretrainLPM(3, 32, train_dataset.n_joints + 1, T=args.t)
    else:
        model = PretrainCoordinateLPM(3, 32, train_dataset.n_joints, T=args.t)
        criterion = CoordinateLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = None
    if args.lr_step_interval:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_interval, gamma=args.gamma)

    summary = None
    if args.host is not None:
        cc = pycrayon.CrayonClient(hostname=args.host)
        if args.experiment in cc.get_experiment_names():
            summary = cc.open_experiment(args.experiment)
        else:
            summary = cc.create_experiment(args.experiment)

    best_acc = 0.
    start_epoch = 0

    if args.checkpoint_name is not None:
        checkpoint = load_checkpoint(model_dir, args.checkpoint_name, model, optimizer)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['accuracy']
        if args.host is not None:
            summary.add_scalar_value('Epoch Valid Accuracy', checkpoint['accuracy'])
            summary.add_scalar_value('Epoch Valid Loss', checkpoint['loss'])

    for epoch in range(start_epoch, args.max_epochs):
        print('\n[epoch ' + str(epoch) + ']')
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, scheduler, args.clip, summary)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)

        if args.host is not None:
            summary.add_scalar_value('Epoch Train Loss', train_loss)
            summary.add_scalar_value('Epoch Valid Loss', valid_loss)
            summary.add_scalar_value('Epoch Train Accuracy', train_acc)
            summary.add_scalar_value('Epoch Valid Accuracy', valid_acc)
            if scheduler is not None:
                summary.add_scalar_value('Learning Rate', scheduler.get_lr()[0])

        is_best = valid_acc >= best_acc
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'accuracy': valid_acc,
                         'loss': valid_loss}, is_best=is_best, checkpoint=model_dir, prefix=start_time_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Architecture
    parser.add_argument('--model', default='hourglass', type=str, choices=['hourglass', 'lpm', 'coord_lpm'])
    parser.add_argument('--t', default=10, type=int)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--block', default='residual', type=str, choices=['residual', 'inverted', 'conv'])

    # Training
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_step_interval', default=None, type=int)
    parser.add_argument('--gamma', default=1, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--decay', default=0, type=float)
    parser.add_argument('--max_epochs', default=5000, type=int)
    parser.add_argument('--resolution', default=256, type=int)
    parser.add_argument('--subset_size', default=None, type=int)
    parser.add_argument('--clip', default=None, type=float)

    # Tensorboard
    parser.add_argument('--experiment', default='Recurrent Stacked Hourglass Training', type=str)
    parser.add_argument('--host', default=None, type=str)

    # Checkpoints
    parser.add_argument('--checkpoint_name', default=None, type=str)

    # Other
    parser.add_argument('--device', default='cpu', type=str, choices=['cpu', '0', '1'])

    main(parser.parse_args())
