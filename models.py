from torch import nn
import torch


class ResNet_Model(nn.Module):
    def __init__(self, opt):
        super(ResNet_Model, self).__init__()
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        self.save_dir = opt.checkpoints_dir
        self.net = init_net(opt)

        if opt.pre_train == 'True':
            self.net = self.net.load_pretrained_model()

        #     加入GPU
        if self.device !='cpu':
            self.net = self.net.to(self.device)


    def forward(self, input):
        self.feat = self.net(input)
        return self.feat


class init_net(nn.Module):
    def __init__(self, opt):
        super(init_net, self).__init__()
        layers = [2, 2, 2, 2]
        self.backbone = ResNet(BasicBlock, layers, opt.num_cls)

    def forward(self, input):
        out = self.backbone(input)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """定义BasicBlock残差块类

        参数：
            inplanes (int): 输入的Feature Map的通道数
            planes (int): 第一个卷积层输出的Feature Map的通道数
            stride (int, optional): 第一个卷积层的步长
            downsample (nn.Sequential, optional): 旁路下采样的操作
        注意：
            残差块输出的Feature Map的通道数是planes*expansion
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, zero_init_residual=False):
        """定义ResNet网络的结构

       参数：
           block (BasicBlock / Bottleneck): 残差块类型
           layers (list): 每一个stage的残差块的数目，长度为4
           num_classes (int): 类别数目
           zero_init_residual (bool): 若为True则将每个残差块的最后一个BN层初始化为零，
               这样残差分支从零开始每一个残差分支，每一个残差块表现的就像一个恒等映射，根据
               https://arxiv.org/abs/1706.02677这可以将模型的性能提升0.2~0.3%
       """

        super(ResNet, self).__init__()
        self.inplanes = 64  # 第一个残差块的输入通道数
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Sequential(
            conv3x3(3, 64, stride=2),
            nn.BatchNorm2d(64),
            conv3x3(64, 64),
            conv3x3(64, 64)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage1 ~ Stage4
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # GAP
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 网络参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """定义ResNet的一个Stage的结构

        参数：
            block (BasicBlock / Bottleneck): 残差块结构
            plane (int): 残差块中第一个卷积层的输出通道数
            blocks (int): 当前Stage中的残差块的数目
            stride (int): 残差块中第一个卷积层的步长
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                nn.AvgPool2d(2, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
