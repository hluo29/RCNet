import torch
import os, time, cv2
import numpy as np
from collections import OrderedDict

from models.network_corres import Network
from util.util import check_path
from util.cal_ssim import SSIM


class Model():
    def __init__(self, opt):
        self.opt          = opt
        self.model_names  = ['G']
        self.loss_names   = ['L1_out1_a', 'L1_out1_b', 'L1_out2_a', 'L1_out2_b', 'L1_out3_a', 'L1_out3_b',
                             'ssim_out1_a', 'ssim_out1_b', 'ssim_out2_a', 'ssim_out2_b', 'ssim_out3_a', 'ssim_out3_b']
        self.visual_names = ['out1_a', 'out1_b', 'out2_a', 'out2_b', 'out3_a', 'out3_b', 'normal_view']
        # self.log_file = os.path.join(self.opt.checkpoints, 'loss_log.txt')
        # with open(self.log_file, 'a') as log_file:
        #     now = time.strftime('%c')
        #     log_file.write('================ Training Loss (%s) ================\n' % now)

        # define networks
        self.netG = Network().cuda()
        # self.get_gradient = ImageGradient(gpu_ids=[0]).cuda()

        # define loss functions
        self.criterion_L1 = torch.nn.L1Loss().cuda()
        self.criterion_ssim = SSIM().cuda()

        # initialize optimizer
        self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.9, 0.999))
        self.lr = self.optimizer.param_groups[0]['lr']

    def set_input(self, low_views, normal_view):
        self.low_views = low_views.cuda()
        self.normal_view = normal_view.cuda()

    def optimize_parameters(self):
        # forward
        self.out1_a, self.out1_b, self.out2_a, self.out2_b, self.out3_a, self.out3_b = self.netG(self.low_views)
        self.optimizer.zero_grad()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        # loss
        self.loss_L1_out1_a = self.criterion_L1(self.out1_a, self.normal_view) 
        self.loss_L1_out1_b = self.criterion_L1(self.out1_b, self.normal_view) 
        self.loss_L1_out2_a = self.criterion_L1(self.out2_a, self.normal_view)
        self.loss_L1_out2_b = self.criterion_L1(self.out2_b, self.normal_view)
        self.loss_L1_out3_a = self.criterion_L1(self.out3_a, self.normal_view) 
        self.loss_L1_out3_b = self.criterion_L1(self.out3_b, self.normal_view) 
        
        self.loss_ssim_out1_a = 1-self.criterion_ssim(self.out1_a, self.normal_view)
        self.loss_ssim_out1_b = 1-self.criterion_ssim(self.out1_b, self.normal_view)
        self.loss_ssim_out2_a = 1-self.criterion_ssim(self.out2_a, self.normal_view)
        self.loss_ssim_out2_b = 1-self.criterion_ssim(self.out2_b, self.normal_view)
        self.loss_ssim_out3_a = 1-self.criterion_ssim(self.out3_a, self.normal_view)
        self.loss_ssim_out3_b = 1-self.criterion_ssim(self.out3_b, self.normal_view)

        # fusion_gradient_x, fusion_gradient_y = self.get_gradient(self.fusion_out)
        # gt_gradient_x, gt_gradient_y = self.get_gradient(self.gt)
        # self.loss_gradient = self.criterion_L1(fusion_gradient_x, gt_gradient_x) + self.criterion_L1(fusion_gradient_y, gt_gradient_y)

        total_loss = self.loss_L1_out1_a + self.loss_L1_out1_b + self.loss_L1_out2_a + self.loss_L1_out2_b + \
            self.loss_L1_out3_a + self.loss_L1_out3_b +\
            self.loss_ssim_out1_a + self.loss_ssim_out1_b + self.loss_ssim_out2_a + self.loss_ssim_out2_b + \
            self.loss_ssim_out3_a + self.loss_ssim_out3_b
        total_loss.backward()
        self.optimizer.step()
        self.lr = self.optimizer.param_groups[0]['lr']
        # torch.cuda.empty_cache()
        
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.out1_a, self.out1_b, self.out2_a, self.out2_b, self.out3_a, self.out3_b = self.netG(self.low_views)
        self.netG.train()

    def get_current_losses(self):
        error_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                error_ret[name] = float(getattr(self, 'loss_' + name))
        return error_ret

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for visual in self.visual_names:
            if isinstance(visual, str):
                visual_ret[visual] = getattr(self, visual)
        return visual_ret

    def print_current_losses(self, epoch, iters, losses, writer, logger, total_iters):
        # message = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        message = '(epoch: %d, iters: %d, lr: %.6f) ' % (epoch, iters, self.lr)
        for k, v in losses.items():
            message += '%s: %.5f ' % (k, v)
            if 'L1' in k:
                writer.add_scalar('Loss_L1/'+k, v, total_iters)
            elif 'ssim' in k:
                writer.add_scalar('Loss_ssim/'+k, v, total_iters)
        logger.info(message)

    def save_networks(self, epoch):
        save_dir = check_path(os.path.join(self.opt.checkpoints, 'model_v2_corres_alpha085_maskavgp_att'))
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(save_dir, save_filename)
                net = getattr(self, 'net'+name)
                if torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda()
                else:
                    torch.save(net.state_dict(), save_path)

    def adjust_learning_rate(self, epoch):
        lr = 0.0002 if epoch <= 200 else 0.00001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr