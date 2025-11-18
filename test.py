from matplotlib.pyplot import title
import torch
import cv2, os, time
import logging
from torchvision import transforms

from options  import Options
from datasets import create_datasets
# from models.network_corres_recur2 import Network
from models.network_corres import Network
from util import util
from skimage.measure import compare_psnr, compare_ssim

from alive_progress import alive_bar


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # test options & logger
    test_opt = Options().parse()
    test_dataset = create_datasets(test_opt)
    dataset_size = len(test_dataset)
    util.setup_logger(test_opt.model, test_opt.logdir, phase='test', screen=True, tofile=True)
    logger = logging.getLogger(test_opt.model)
    logger.info('Load over')

    with torch.no_grad():
        model_net = Network().cuda()
        model_net.load_state_dict(torch.load('/home/cbl/LUOHAO/RCNet/pretrained_model/pretrained_model_MVLT.pth'))
        # model_v1_corres_alpha085_maskavgp
        value_psnr = 0
        value_ssim = 0
        model_net.eval()
        if torch.cuda.device_count() > 1:
            model_net = torch.nn.DataParallel(model_net)
        testing_time_start = time.time()
        with alive_bar(dataset_size, title='test', spinner='classic') as bar:
            for i, data in enumerate(test_dataset):
                iter_data_time = time.time()
                out1_a, out1_b, out2_a, out2_b, out3_a, out3_b = model_net(data['low_views'].cuda())
                iter_data_time = time.time()-iter_data_time
                out = util.tensor2img(torch.clamp(out3_b, 0, 1))
                gt  = util.tensor2img(data['normal_view'])
                # cv2.imwrite(os.path.join(test_opt.resultdir, 'Ours_v1_ill', data['img_name'][0]), ill)

                calc_psnr = compare_psnr(out, gt)
                calc_ssim = compare_ssim(out, gt, multichannel=True)
                value_psnr += calc_psnr
                value_ssim += calc_ssim
                # logger.info('Idx: %d, psnr: %.5f, ssim: %.5f, image_name: %s' % (i, calc_psnr, calc_ssim, data['img_name'][0]))
                # torch.cuda.empty_cache()
                bar()

    logger.info('PSNR= %.5f, SSIM= %.5f' % (value_psnr/dataset_size, value_ssim/dataset_size))
    # testing_time = time.time()-testing_time_start
    # logger.info('Time: %.4f, Average: %.4f' % (testing_time, testing_time / float(dataset_size)))
    # eval_file = os.path.join(opt.checkpoints, 'eval_Rain100H.txt')
    # with open(eval_file, "a+") as eval_file:
    #     eval_file.write('%d_net_G.pth load over\n' % m)
    #     eval_file.write('PSNR= %.5f, SSIM= %.5f\n' % (value_psnr / float(dataset_size), value_ssim / float(dataset_size)))
    #     eval_file.write('\n')
    del model_net
    torch.cuda.empty_cache()
    time.sleep(5)
        


