import os, time, cv2
from tkinter import image_names
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import logging

from options import Options
from datasets import create_datasets
from models.model_corres import Model
from util import util
from skimage.measure import compare_psnr, compare_ssim

from tensorboardX import SummaryWriter
from alive_progress import alive_bar
# from ipdb import set_trace

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # train/test options & logger
    train_opt = Options().parse(True)
    test_opt = Options().parse()
    util.setup_logger(train_opt.model, train_opt.logdir, phase='train', screen=True, tofile=True)
    logger = logging.getLogger(train_opt.model)
    # set_trace()
    # datasets & model
    training_dataset = create_datasets(train_opt)
    dataset_size = len(training_dataset)
    test_dataset = create_datasets(test_opt)
    model = Model(train_opt)
    # model.netG.load_state_dict(torch.load(''))
    # util.clcfiles(train_opt.logdir, train_opt.model)
    logger.info('Load over')

    total_iters = 0
    best_psnr = 0
    writer = SummaryWriter(util.check_path(os.path.join(train_opt.logdir, train_opt.model)))
    # training
    for epoch in range(train_opt.epoch_count, train_opt.niter + 1):
        torch.cuda.empty_cache()
        epoch_start_time = time.time()
        epoch_iter = 0
        model.adjust_learning_rate(epoch)
        for i, data in enumerate(training_dataset):
            iter_start_time = time.time()
            total_iters += 1
            epoch_iter += 1

            model.set_input(data['low_views'], data['normal_view'])
            model.optimize_parameters()

            if total_iters % train_opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / train_opt.batch_size
                model.print_current_losses(epoch, epoch_iter, losses, writer, logger, total_iters)

            if epoch_iter % train_opt.visual_freq == 0:
                visuals = model.get_current_visuals()
                for label, image in visuals.items():
                    image = util.tensor2img(torch.clamp(image, 0, 1))
                    image_name = '%d-%d-%s-%s' % (epoch, i, label, data['img_name'][0])
                    cv2.imwrite(os.path.join(util.check_path(train_opt.savedir + train_opt.model), image_name), image)
                    # util.save_image(image, os.path.join(opt.savedir, opt.model))
            # torch.cuda.empty_cache()
        
        # test
        if epoch % train_opt.test_freq == 0:
            value_psnr = 0
            value_ssim = 0
            with alive_bar(len(test_dataset), title='Test', spinner='classic') as bar:
                for i, data in enumerate(test_dataset):
                    model.set_input(data['low_views'], data['normal_view'])
                    model.test()
                    
                    visuals = model.get_current_visuals()
                    out = visuals['out3_b'] # out3
                    normal = visuals['normal_view']
                    out = util.tensor2img(torch.clamp(out, 0, 1))
                    normal = util.tensor2img(torch.clamp(normal, 0, 1))
                    cv2.imwrite(os.path.join(util.check_path(train_opt.resultdir + train_opt.model), data['img_name'][0]), out)
                    
                    # calculate PSNR & SSIM
                    calc_psnr = compare_psnr(out, normal)
                    calc_ssim = compare_ssim(out, normal, multichannel=True)
                    value_psnr += calc_psnr
                    value_ssim += calc_ssim
                    bar()
                logger.info('Epoch: %d, PSNR: %.5f, SSIM: %.5f' % (epoch, value_psnr/len(test_dataset), value_ssim/len(test_dataset)))

        if epoch % train_opt.save_epoch_freq == 0:
            logger.info('Saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)
            model.save_networks('latest')
        logger.info('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, train_opt.niter, time.time() - epoch_start_time))
        torch.cuda.empty_cache()
    logger.info('model_v2_corres_alpha085_maskavgp_att')
    # model_v1_corres_alpha085_maskavgp