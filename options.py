import argparse

class Options():
    """This class defines options used during training and test time."""
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # data
        parser.add_argument('--dataroot',    default='/home/cbl/LUOHAO/Datasets/MVLT')
        parser.add_argument('--checkpoints', default='./checkpoints')
        parser.add_argument('--savedir',     default='./checkpoints/savedir/')
        parser.add_argument('--logdir',      default='./checkpoints/logdir/')
        parser.add_argument('--resultdir',   default='./checkpoints/result/')

        # training and testing
        parser.add_argument('--epoch_count', type=int,  default=1)
        parser.add_argument('--niter',       type=int,  default=500)
        parser.add_argument('--batch_size',  type=int,  default=6)
        parser.add_argument('--mode',        type=bool, default=True, help='True: train, False: test')
        parser.add_argument('--model',       type=str,  default='ours_v2_corres_alpha085_maskavgp_att')
        # ours_v1_corres_alpha085_maskavgp

        # visualization and test
        parser.add_argument('--print_freq',       type=int, default=5)
        parser.add_argument('--visual_freq',      type=int, default=5)
        parser.add_argument('--test_freq',        type=int, default=510)
        parser.add_argument('--save_epoch_freq',  type=int, default=1)
        parser.add_argument('--save_latest_freq', type=int, default=2)

        self.initialized = True
        self.isTrain = True
        return parser

    def parse(self, train_mode=False):
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)
        self.parser = parser
        opt = parser.parse_args()
        opt.isTrain = self.isTrain
        
        if not train_mode:
            opt.mode = False
        self.opt = opt
        return opt


if __name__ == '__main__':

    opt = Options().parse()
    print(True)