import os
import random
import time

import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# from util import AverageMeter
import models
from dataset import CustomDataset
from options import BaseOptions
from models import ResNet_Model
from tensorboardX import SummaryWriter


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_torch(seed=2019):
    random.seed(seed)  # python
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False  # cuda  conv
    torch.backends.cudnn.deterministic = True


# set seeds
seed_torch(2019)


def train(opt, trainloader, model, criterion, optimizer, epoch, epoch_step, writer):
    """

    :param opt: options
    :param trainloader: train dataset
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:--new epoch
    :param epoch_step:--total each epoch num
    :param writer:--tensorboard
    :return:
    """

    logFile_name_train = os.path.join(opt.log_dir, 'train_log.txt')
    log_file=open(logFile_name_train, "a")
    loss = 0
    Epoch = opt.end_epoch
    cuda = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device(
        'cpu')  # get device name: CPU or GPU
    print("start train!!")

    # start train
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    now = time.strftime("%m %d %H-%M")
    log_file.write('================ training time (%s) ================\n' % now)
    with tqdm(total=epoch_step, desc=f'Epoch {epoch}/{Epoch}', postfix=dict, mininterval=1.5) as pbar:
        end = time.time()
        for iteration, data in enumerate(trainloader):
            if iteration >= epoch_step:
                break

            images, targets = data[0], data[1]

            if cuda:
                images = images.cuda()
                # print(images.requires_grad)
                targets = np.array(targets, dtype=int)
                targets = torch.from_numpy(targets).type(torch.Tensor).cuda()
                # print(targets,type(targets))

            optimizer.zero_grad()
            outputs = model(images)
            # print(outputs.shape)

            loss = criterion(outputs, targets.long())
            # print(loss.requires_grad)
            losses.update(loss.item(), images.size(0))

            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)

            loss += loss.item()

            pbar.set_postfix(**{'loss': loss.item() / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
        if epoch % opt.save_epoch == 0:
            print("\nThis is %s epoch %s iter save model new!" % (epoch, iteration))
            # model.save_networks(epoch, loss.item())
            save_path = opt.checkpoints_dir + '/%s_net_L_%3f_2.0.pth' % (epoch, loss)
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, save_path)

        # print('\nlabel:%3s,predict:%3s' % (targets, outputs.argmax(-1)))
        # print(batch_time)
        # print('Batch_Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time))
        log_file.write('label:%3s,predict:%3s\n' % (targets, outputs.argmax(-1)))
        log_file.write('Batch_Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'.format(batch_time=batch_time))
        writer.add_scalar('loss/train_loss', losses.val, global_step=epoch)


def val(opt, valloader, model, criterion,  epoch, epoch_step, writer):
    """

        :param opt: options
        :param trainloader: train dataset
        :param model:
        :param criterion:
        :param optimizer:
        :param epoch:--new epoch
        :param epoch_step:--total each epoch num
        :param writer:--tensorboard
        :return:
        """
    print("start val!")
    logFile_name_val = os.path.join(opt.log_dir, 'val_log.txt')
    log_file=open(logFile_name_val, "a")
    val_loss = 0
    val_losses = AverageMeter()
    accuray = AverageMeter()
    Epoch = opt.end_epoch
    cuda = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if (opt.gpu_ids) else torch.device('cpu')
    now = time.strftime("%m %d %H-%M")
    model.eval()
    log_file.write('================ valing time %s ================\n' %now)
    with tqdm(desc=f'Epoch {epoch}/{Epoch}', total=epoch_step, postfix=dict, miniters=1.5) as pbar:
        for iteration , data in enumerate(valloader):

            with torch.no_grad():
                images, targets = data[0], data[1]

                if cuda:
                    images = images.cuda()
                    targets_np = np.array(targets, dtype=int)
                    targets = torch.from_numpy(targets_np).type(torch.Tensor).cuda()

                pred = model(images)
                val_loss = criterion(pred, targets.long())

                val_losses.update(val_loss.item(), targets.size(0))
                pred = pred.argmax(-1)
                # print(pred,targets)
                accuray.update((pred == targets).sum() / opt.batch_size, targets.size(0))

                val_loss += val_loss.item()

                # pbar.set_postfix(**{'loss': loss / (iteration + 1),
                #                     'lr': get_lr(optimizer)})
                pbar.set_postfix(**{'loss': val_loss.item()/(iteration + 1)})
                pbar.update(1)

        print("Accuracy {accuray.val:.2f}".format(accuray=accuray))
        log_file.write("Accuracy {accuray.val:.2f}\n".format(accuray=accuray))
        writer.add_scalar('loss/val_loss', val_losses.val, global_step=epoch)


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    # -------------------------------------------- step 1/4 : 加载数据 ---------------------------
    # opt       dataroot val_dataroot  transform
    opt = BaseOptions().parse()
    trainset = CustomDataset(opt.dataroot, isTrain=True)
    valset = CustomDataset(opt.val_dataroot)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size_val,
                                            shuffle=True, num_workers=2)
    dataset_size = len(trainset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    dataset_size_val = len(valset)  # get the number of images in the dataset.
    print('The number of validation images = %d' % dataset_size_val)

    # -------------------------------------------- step 2/4 : 定义网络 ---------------------------
    model = ResNet_Model(opt)

    # ------------------------------------ step 3/4 : 定义损失函数和优化器等 -------------------------
    criterion = nn.CrossEntropyLoss()  # 选择损失函数
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)  # 选择优化器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 设置学习率下降策略
    writer = SummaryWriter('run/ResNet18')

    # ------------------------------------ step 4/4 : 训练 -----------------------------------------
    # get total each epoch num
    epoch_step = dataset_size // opt.batch_size
    epoch_step_val = dataset_size_val // opt.batch_size_val
    start_epoch = opt.start_epoch
    now = time.strftime("%m %d %H-%M")

    if opt.continue_work:
        load_path = opt.checkpoints_dir + '/' + opt.load_name
        checkpoints = torch.load(load_path)
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        start_epoch = checkpoints['epoch']
        loss = checkpoints['loss']



    for epoch in range(start_epoch, opt.end_epoch):

        train(opt, trainloader, model, criterion, optimizer, epoch, epoch_step, writer)

        val(opt, valloader, model, criterion, epoch, epoch_step_val, writer)
        #         each epoch change train
        lr_scheduler.step()

    print('Finish')
