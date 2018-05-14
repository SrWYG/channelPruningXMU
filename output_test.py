import argparse
from xmumodel.edsr_deconv import EDSR
from xmudata.preddata import data_for_predict
import tensorflow as tf
import tensorlayer as tl
import sys
from xmuutil import utils
import numpy as np
import os
import time
from tqdm import tqdm

FLAGS=None


def enhance_predict(lr_imgs, network=None):
    outs_list = []
    for _, flip_axis in enumerate([0, 1, 2, -1]):
        for _, rotate_rg in enumerate([0, 90]):
            en_imgs = utils.enhance_imgs(lr_imgs,  rotate_rg, flip_axis)
            outs = network.predict(en_imgs)
            anti_outs = utils.anti_enhance_imgs(outs, rotate_rg, flip_axis)
            outs_list.append(anti_outs)
    return np.mean(outs_list, axis=0)


def main(_):
    if not os.path.exists(FLAGS.outdir):
        os.mkdir(FLAGS.outdir)
    if(os.path.exists(FLAGS.prunedlist_path)):
        prunedlist = np.loadtxt(FLAGS.prunedlist_path,dtype=np.int64)
    else:
        prunedlist = [0]*16

    #network = EDSR(FLAGS.layers, FLAGS.featuresize, FLAGS.scale, FLAGS.channels)
    network = EDSR(FLAGS.layers, FLAGS.featuresize, FLAGS.scale,FLAGS.channels, FLAGS.channels, prunedlist)
    network.buildModel()
    network.resume(FLAGS.reusedir, 1999)

    hr_list, lr_imgs, groundtruth_imgs = data_for_predict(FLAGS.datadir, FLAGS.groundtruth, FLAGS.postfixlen)

    if groundtruth_imgs:
        psnr_list = []
        time_list = []
        fo = open(FLAGS.outdir + '/psnr.csv', 'w')
        fo.writelines("file, PSNR\n")
        for lr_img, groundtruth_img, hr_name in zip(lr_imgs, groundtruth_imgs, hr_list):
            start = time.time()
            out = network.predict([lr_img])
            # out = enhance_predict([lr_img],network)
            use_time = time.time()-start
            time_list.append(use_time)
            tl.vis.save_image(out[0], FLAGS.outdir + '/' + hr_name)
            psnr = utils.psnr_np(groundtruth_img, out[0], scale=8)
            print('%s : %.6f' % (hr_name, psnr))
            psnr_list.append(psnr)
            fo.writelines("%s, %.6f\n" % (hr_name, psnr))

        print(np.mean(psnr_list))
        print(np.mean(time_list))
        fo.writelines("%d, Average,0, %.6f" % (-1, np.mean(psnr_list)))
        fo.close()

    else:
        for i in tqdm(range((len(hr_list)))):
            out = network.predict([lr_imgs[i]])
            tl.vis.save_image(out[0], FLAGS.outdir + '/' + hr_list[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", default='data/DIV2K_2018/DIV2K_valid_LR_x8')
    parser.add_argument("--groundtruth",default='data/DIV2K/DIV2K_valid_HR')
    parser.add_argument("--postfixlen", default=2,type=int)
    parser.add_argument("--scale",default=8,type=int)
    parser.add_argument("--layers",default=16,type=int)
    parser.add_argument("--featuresize",default=128,type=int)
    parser.add_argument("--reusedir",default='prune_ckpt/futune/v1_48')
    parser.add_argument("--outdir", default='out_test/futune/v1_48')
    parser.add_argument("--channels",default=3,type=int)
    parser.add_argument("--prunedlist_path", default="prune_ckpt/pruned/v1_48/prunedlist")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
