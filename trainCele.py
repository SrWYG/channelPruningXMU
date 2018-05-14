from xmudata.DIV2K2018 import DIV2K2018
#from xmudata.adddata import AddData
import argparse
#from xmumodel.cyclesr import CycleSR
from xmumodel.edsr_deconv_T import EDSR
from xmumodel.edsr_deconv_nv_prune_dec import EDSR_D
from xmumodel.edsr_deconv_la_nodec import EDSR_T
import tensorflow as tf
import sys
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

FLAGS=None

def main(_):
    data = DIV2K2018(FLAGS.groundtruthdir, FLAGS.datadir, None, None,
                     FLAGS.imgsize, FLAGS.scale, FLAGS.postfixlen, FLAGS.postfixlen)
    
    if(os.path.exists(FLAGS.prunedlist_path)):
        prunedlist = np.loadtxt(FLAGS.prunedlist_path,dtype=np.int64)
    else:
        prunedlist = [0]*19  
    #print("========================"+FLAGS.prune) 
    if(FLAGS.prune):
        network = EDSR_T(FLAGS.layers, FLAGS.featuresize, FLAGS.scale,FLAGS.channels, FLAGS.channels, prunedlist, FLAGS.prunesize)
    else:
        network = EDSR(FLAGS.layers, FLAGS.featuresize, FLAGS.scale,FLAGS.channels, FLAGS.channels, prunedlist, FLAGS.prunesize)
    #network = CycleSR(FLAGS.featuresize, FLAGS.layers, FLAGS.channels)
    network.buildModel()
    network.set_data(data)
    network.train(FLAGS.batchsize, FLAGS.iterations, FLAGS.lr_init, FLAGS.lr_decay, FLAGS.decay_every,
                  FLAGS.savedir, True, FLAGS.reusedir,167500, log_dir=FLAGS.logdir)
    #network.train(FLAGS.batchsize, 
    #              FLAGS.savedir, True, FLAGS.reusedir, 500, log_dir=FLAGS.logdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--groundtruthdir",default="data/cele/celeba")
    """
    datadir                               postfix_len scale  track
    data/DIV2K_2018/DIV2K_train_LR_x8          2        8    1: bicubic downscaling x8 competition
    data/DIV2K_2018/DIV2K_train_LR_mild        3        4    2: realistic downscaling x4 with mild conditions competition
    data/DIV2K_2018/DIV2K_train_LR_difficult   3        4    3: realistic downscaling x4 with difficult conditions competition
    data/DIV2K_2018/DIV2K_train_LR_wild        4        4    4: wild downscaling x4 competition
    """
    parser.add_argument("--datadir",default="data/cele/celeba_x8")
    parser.add_argument("--valid_groundtruthdir", default='data/cele/valid')
    parser.add_argument("--valid_datadir", default="data/cele/valid_x8")
    parser.add_argument("--postfixlen",default=0)
    parser.add_argument("--imgsize",default=16,type=int)
    parser.add_argument("--scale",default=8,type=int)
    parser.add_argument("--layers",default=16,type=int)
    parser.add_argument("--featuresize",default=128,type=int)
    parser.add_argument("--channels", default=3, type=int)
    parser.add_argument("--batchsize",default=16,type=int)
    parser.add_argument("--savedir",default='ckpt_cele_2/')
    parser.add_argument("--logdir", default='log_cele_2/')
    parser.add_argument("--reusedir",default='ckpt_cele')
    parser.add_argument("--psnrpath", default='out/psnr_v6000_train.csv')
    #parser.add_argument("--savedir", default='result/track4/cyclesr/ckpt')
    #parser.add_argument("--logdir", default='result/track4/cyclesr/log')
    #parser.add_argument("--reusedir",default='result/track2/22_749_v500_cyclegan_v3/ckpt')
    #parser.add_argument("--psnrpath", default='result/track2/22_749_v500_cyclegan_v3/out/psnr_v6000_train.csv')
    parser.add_argument("--iterations",default=1000000,type=int)
    parser.add_argument("--lr_init", default=1e-4)
    parser.add_argument("--lr_decay", default=0.5)
    parser.add_argument("--decay_every", default=50000)
    parser.add_argument("--prunedlist_path", default="no")
    #prunedlist_path= "prune_ckpt/pruned/v1_4/prunedlist"
    #if(os.path.exists(prunedlist_path)):
    #    prunedlist = np.loadtxt(prunedlist_path,dtype=np.int64)
    #else:
    #    prunedlist = [0]*16    
    #parser.add_argument("--prunedlist", default=prunedlist)
    parser.add_argument("--prunesize",type=int, default=10)
    parser.add_argument("--prune",default=False,type=bool)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
