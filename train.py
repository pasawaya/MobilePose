
import time
import argparse
import pycrayon
import configargparse
import torch.nn.utils as utils
from tqdm import tqdm
from datasets.MPII import MPII
from datasets.LSP import LSP
from datasets.PennAction import PennAction

from torch.utils.data import DataLoader

from utils.train_utils import *
from utils.augmentation import *
from utils.dataset_utils import *
from utils.evaluation import accuracy

from models.RecurrentStackedHourglass import PretrainRecurrentStackedHourglass
from models.LSTMPoseMachine import LPM
from models.CoordinatePoseMachine import CoordinateLPM
from models.modules.ResidualBlock import ResidualBlock
from models.modules.InvertedResidualBlock import InvertedResidualBlock
from models.modules.ConvolutionalBlock import ConvolutionalBlock
from models.losses.MSESequenceLoss import MSESequenceLoss
from models.losses.CoordinateLoss import CoordinateLoss


def train(model, loader, criterion, optimizer, device, r, scheduler=None, clip=None, summary=None, debug=False):
    loss_avg = RunningAverage()
    acc_avg = RunningAverage()
    time_avg = RunningAverage()

    model.train()

    with tqdm(total=len(loader)) as t:
        for i, (frames, labels, centers, meta, unnormalized) in enumerate(loader):
            frames, labels, centers, meta = frames.to(device), labels.to(device), centers.to(device), meta.to(device)

            start = time.time()
            outputs = model(frames, centers)
            time_avg.update(time.time() - start)

            if debug:
                visualize(unnormalized, labels, outputs)

            if isinstance(criterion, CoordinateLoss):
                loss = criterion(*outputs, meta, device)
                acc = coord_accuracy(outputs[1], meta, r=r)
            else:
                loss = criterion(outputs, labels)
                acc = accuracy(outputs, labels, r=r)

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
                          loss_avg='{:05.3f}'.format(loss_avg()), acc_avg='{:05.3f}%'.format(acc_avg() * 100),
                          time_avg='{:05.3f}s'.format(time_avg()))
            t.update()

        return loss_avg(), acc_avg()


def validate(model, loader, criterion, device, r):
    loss_avg = RunningAverage()
    acc_avg = RunningAverage()

    model.eval()
    for i, (frames, labels, centers, meta, _) in enumerate(loader):
        frames, labels, centers, meta = frames.to(device), labels.to(device), centers.to(device), meta.to(device)

        outputs = model(frames, centers)
        if isinstance(criterion, CoordinateLoss):
            loss = criterion(*outputs, meta, device)
            acc = coord_accuracy(outputs[1], meta, r=r)
        else:
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels, r=r)

        loss_avg.update(loss.item())
        acc_avg.update(acc)

    return loss_avg(), acc_avg()


def main(args):
    start_time_prefix = str(int(time.time()))[-4:] + "_"
    print('Checkpoint prefix will be ' + start_time_prefix)

    device_name = 'cpu' if args.gpu is None else 'cuda:' + str(args.gpu)
    device = torch.device(device_name)
    loader_args = {'num_workers': 1, 'pin_memory': True} if 'cuda' in device_name else {}

    if args.dataset == 'penn':
        dataset_name = 'PennAction'
        dataset = PennAction
        transformer = VideoTransformer
    elif args.dataset == 'MPII':
        dataset_name = 'MPII'
        dataset = MPII
        transformer = ImageTransformer
    else:
        dataset_name = 'LSP'
        dataset = LSP
        transformer = ImageTransformer
    root = os.path.join(args.data_dir, dataset_name)

    mean_name = 'means.npy'
    mean_path = os.path.join(root, mean_name)
    if not os.path.isfile(mean_path):
        temp = dataset(args.t, root=root, transformer=None, output_size=args.resolution, train=True)
        mean, std = compute_mean(temp)
        np.save(mean_path, np.array([mean, std]))

    mean, std = np.load(os.path.join(root, 'means.npy'))
    train_transformer = transformer(output_size=args.resolution, mean=mean, std=std)
    valid_transformer = transformer(output_size=args.resolution,
                                    p_scale=0.0, p_flip=0.0, p_rotate=0.0,
                                    mean=mean, std=std)

    stride = 4 if args.model == 'hourglass' else 8
    offset = 0 if args.model == 'hourglass' else -1
    include_background = args.model != 'coord_lpm'
    train_dataset = dataset(args.t, root=root, transformer=train_transformer, output_size=args.resolution,
                            train=True, subset_size=args.subset_size, sigma=7, stride=stride, offset=offset,
                            include_background=include_background)
    valid_dataset = dataset(args.t, root=root, transformer=valid_transformer, output_size=args.resolution,
                            train=False, subset_size=args.subset_size, sigma=7, stride=stride, offset=offset,
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
        model = LPM(3, 32, train_dataset.n_joints + 1, device, T=args.t)
    else:
        model = CoordinateLPM(3, 32, train_dataset.n_joints, device, T=args.t)
        criterion = CoordinateLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.step_size:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)

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
        checkpoint = load_checkpoint(args.model_dir, args.checkpoint_name, model, optimizer)
        start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch, args.max_epochs):
        print('\n[epoch ' + str(epoch) + ']')
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, args.pck_r, scheduler,
                                      args.clip, summary, args.debug)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device, args.pck_r)

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
                         'optimizer': optimizer.state_dict()},
                        is_best=is_best, checkpoint=args.model_dir, prefix=start_time_prefix)
        best_acc = valid_acc if is_best else best_acc


if __name__ == '__main__':
    parser = configargparse.ArgParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add('--config', is_config_file=True, help='config file path')

    # Architecture
    parser.add('--model', default='hourglass', type=str, choices=['hourglass', 'lpm', 'coord_lpm'], help='model type')
    parser.add('--t', default=10, type=int, help='length of input sequences (i.e. # frames)')
    parser.add('--depth', default=4, type=int, help='depth of each hourglass module')
    parser.add('--block', default='res', type=str, choices=['res', 'inv', 'conv'], help='type of hourglass blocks')

    # Training
    parser.add('--lr', default=1e-3, type=float, help='base learning rate')
    parser.add('--step_size', default=None, type=int, help='period of learning rate decay')
    parser.add('--gamma', default=1, type=float, help='multiplicative factor of learning rate decay')
    parser.add('--batch_size', default=4, type=int, help='training batch size')
    parser.add('--weight_decay', default=0, type=float, help='l2 decay coefficient')
    parser.add('--max_epochs', default=5000, type=int, help='maximum training epochs')
    parser.add('--resolution', default=256, type=int, help='model input image resolution')
    parser.add('--subset_size', default=None, type=int, help='size of training subset (for sanity overfitting)')
    parser.add('--clip', default=None, type=float, help='maximum norm of gradients')

    # Tensorboard
    parser.add('--experiment', default='Pose Estimation Training', type=str, help='name of Tensorboard experiment')
    parser.add('--host', default=None, type=str, help='Tensorboard host name')

    # Checkpoints
    parser.add('--checkpoint_name', default=None, type=str, help='checkpoint file name in experiments/ to resume from')
    parser.add('--model_dir', default='experiments', type=str, help='directory to store/load checkpoints from')

    # Other
    parser.add('--data_dir', default='data', type=str, help='directory containing data')
    parser.add('--gpu', default=None, type=int, help='gpu id to perform training on')
    parser.add('--pck_r', default=0.2, type=float, help='r coefficient for pck computation')
    parser.add('--dataset', default='penn', type=str, choices=['penn', 'mpii', 'lsp'], help='dataset to train on')
    parser.add('--debug', action='store_true', help='visualize model inputs and outputs')

    main(parser.parse_args())
