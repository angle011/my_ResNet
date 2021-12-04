import os
import time

import cv2
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim

from dataset import CustomDataset
from models import ResNet_Model
from options import BaseOptions
from train import val
import matplotlib.pyplot as plt


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


def tes_time():
    import time
    print(time.strftime("%m/%d/%H-%M"), type(time.strftime("%m/%d/%H-%M")))


def tes_Average():
    data = AverageMeter()
    for i in range(10):
        data.update(i)
    print(data.avg)


def tes_cuda():
    x = torch.rand((3, 3))
    # print(type(x))
    print(x.cuda(0))
    # print(x.item(), type(x.item()))


def _state_load_dict():
    opt = BaseOptions().parse()
    from models import ResNet_Model
    test_model = ResNet_Model(opt)
    test_model.load_state_dict(torch.load('/home/liu1227/Download/my_ResNet/checkpoints/30_net_L_1.358699.pth'),
                               strict=False)
    # print(test_model)
    param = test_model.state_dict()
    # for k,v in param.items():
    #     print('%s:  %s'%(k,v.shape))
    with open('./log/test_model.txt', 'a') as f:
        f.write('\n\n')
        f.write(str(test_model))
    with open('./log/test_model_key_value.txt', 'a') as f1:
        f1.write('\n\n')
        for k, v in param.items():
            f1.write('%s:  %s\n' % (k, v))

    pass


def val_demo():
    # -------------------------------------------- step 1/4 : 加载数据 ---------------------------
    # opt       dataroot val_dataroot  transform
    opt = BaseOptions().parse()
    valset = CustomDataset(opt.val_dataroot)
    valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size_val,
                                            shuffle=True, num_workers=2)
    dataset_size_val = len(valset)  # get the number of images in the dataset.
    print('The number of validation images = %d' % dataset_size_val)

    # -------------------------------------------- step 2/4 : 定义网络 ---------------------------
    model = ResNet_Model(opt)

    # ------------------------------------ step 3/4 : 定义损失函数和优化器等 -------------------------
    criterion = nn.CrossEntropyLoss()  # 选择损失函数
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)  # 选择优化器
    writer = SummaryWriter('run/ResNet18')

    # ------------------------------------ step 4/4 : 训练 -----------------------------------------
    # get total each epoch num
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

        # load_net_old(model)
        # # model_urls这个字典是预训练模型的下载地址
        # model_urls = {
        #     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        #     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        #     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        #     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        #     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        # }
        # import torch.utils.model_zoo as model_zoo
        # pretrain_dict = model_zoo.load_url(model_urls['resnet18'])
        # now_dict =getattr(model,'net')
        # now_dict=getattr(now_dict,'backbone')
        # now_dict=now_dict.state_dict()
        # pretrained_dict = {k: v for k, v in pretrain_dict.items() if (k in now_dict and 'fc' not in k)}
        # now_dict.update(pretrained_dict)
        #
        # model.net.backbone.load_state_dict(now_dict,strict=False)
        model.cuda(0)

        # compare the old one and the pretrained  differences
        # f=open('./log/now_dict.txt','a')
        # for k,v in now_dict.items():
        #     f.write('%s:  %s\n' %(k,v))
        # f.close()
        # f1 = open('./log/pre_dict.txt', 'a')
        # for k, v in pretrain_dict.items():
        #     f1.write('%s:  %s\n' % (k, v))
        # f1.close()

    # _state_load_dict()

    # logFile_name_val = os.path.join(opt.log_dir, now + 'val_log_demo.txt')
    # log_file_val = open(logFile_name_val, "a")

    for epoch in range(start_epoch, opt.end_epoch):
        val(opt, valloader, model, criterion, epoch, epoch_step_val, writer)

    print('Finish')


def load_net_old(model):
    """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
    load_path = './checkpoints/30_net_L_1.358699.pth'
    # by attr name to search net
    net = getattr(model, 'net')
    # net= getattr(net,'backbone')
    print('loading the model from %s' % load_path)
    state_dict = torch.load(load_path)
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    net.load_state_dict(state_dict, strict=False)


def sreach_cls_num():
    file_cls_num=[]
    path = '/data/adv/train'
    list_dir = os.walk(path)
    for root, dirs, files in list_dir:
        dirs=list(map(int,dirs))
        dirs.sort()
        for dir in dirs:
            PATH = os.path.join(root, str(dir))
            # [print('{:^8d}is{:^10d}'.format(dir, _files.__len__())) for _, __, _files in os.walk(PATH)]
            for _, __, _files in os.walk(PATH):
                file_cls_num.append(_files.__len__())

    plt.figure( figsize=(35,8))
    print( len(file_cls_num))
    plt.bar(range(137),file_cls_num, align='center')
    plt.show()
    pass


if __name__ == "__main__":
    # torch.cuda.empty_cache()

    tes_cuda()
    # _state_load_dict()
    # val_demo()
    # sreach_cls_num()

    pass
