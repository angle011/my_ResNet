import argparse
import os
from util import util
import torch
import models


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        # parser.add_argument('--dataroot', type=str, default='./LEVIR-CD', help='path to images (should have subfolders A, B, label)')
        # parser.add_argument('--val_dataroot', type=str, default='./LEVIR-CD', help='path to images in the val phase (should have subfolders A, B, label)')
        parser.add_argument('--dataroot', type=str, default='/data/adv/train.txt',help='path to images (txt_path)')
        parser.add_argument('--val_dataroot', type=str, default='/data/adv/val.txt',help='path to images in the val phase (should have subfolders A, B, label)')
        parser.add_argument('--name', type=str, default='myResNet', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--log_dir',type=str,default='./log')
        # net papmeters
        parser.add_argument('--in_c',type=int,default=3,help='init channels for image')
        parser.add_argument('--out_c', type=int, default=64, help='init channels for image')
        # parser.add_argument('--block',type=str,default='BasicBlock',help='block (BasicBlock / Bottleneck): 残差块结构')
        parser.add_argument('--num_cls',type=int,default=137,help='the number of item class')
        parser.add_argument('--continue_work',action='store_true',help='continue_work')
        parser.add_argument('--load_name',type=str,default='8_net_L_8.137641_2.0.pth',help='you can select which model to load')


        # # model parameters
        parser.add_argument('--model', type=str, default='ResNet18', help='chooses which model to use. [CDF0 | CDFA]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB ')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB')
        parser.add_argument('--pre_train', type=str, default='False', help='need load pre train model')
        parser.add_argument('--f_c', type=int, default=64, help='feature extractor channel num')
        parser.add_argument('--n_class', type=int, default=2, help='# of output pred channels: 2 for num of classes')
        parser.add_argument('--lr',type=float,default=1,help='start learning rate')

        # additional parameters
        parser.add_argument('--start_epoch',type=int,default=1,help='the start point epoch')
        parser.add_argument('--end_epoch',type=int,default=100,help='the end point epoch')
        parser.add_argument('--batch_size', type=int, default=12, help='input batch size')
        parser.add_argument('--batch_size_val', type=int, default=70, help='input batch size')
        parser.add_argument('--save_epoch',type=int, default=2,help="every * epoch ,it will save epoch !")

        self.initialized = True
        return parser

    # def gather_options(self):
    #     """Initialize our parser with basic options(only once).
    #     Add additional model-specific and dataset-specific options.
    #     These options are defined in the <modify_commandline_options> function
    #     in model and dataset classes.
    #     """
    #     if not self.initialized:  # check if it has been initialized
    #         parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #         parser = self.initialize(parser)
    #
    #     # get the basic options
    #     opt, _ = parser.parse_known_args()
    #
    #     # modify model-related parser options
    #     model_name = opt.model
    #     model_option_setter = models.get_option_setter(model_name)
    #     parser = model_option_setter(parser, self.isTrain)
    #     opt, _ = parser.parse_known_args()  # parse again with new defaults
    #
    #     # modify dataset-related parser options
    #     dataset_name = opt.dataset_mode
    #     if dataset_name != 'concat':
    #         dataset_option_setter = data.get_option_setter(dataset_name)
    #         parser = dataset_option_setter(parser, self.isTrain)
    #
    #     # save and return the parser
    #     self.parser = parser
    #     return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # util.mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, 'opt.txt')
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        self.parser = parser
        opt= parser.parse_args()

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
if __name__=="__main__":
    print(torch.cuda.is_available())