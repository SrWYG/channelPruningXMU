from xmumodel.model import Model
from xmuutil import utils
from xmuutil.relulayer import ReluLayer
import tensorflow as tf
import tensorlayer.layers as tl
from xmuutil.custom_vgg16 import loadWeightsData,custom_Vgg16

"""
An implementation of EDSR used for
super-resolution of images as described in:

`Image Super-Resolution Using Dense Skip Connections`
(http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)

"""

class DenseNet(Model):
    def __init__(self,dense_block=8,growth_rate=16,bottleneck_size=256,num_layers=8,feature_size=16,scale=8,output_channels=3,is_subpixel=True,is_bn=False):
        Model.__init__(self,num_layers,feature_size,scale,output_channels)

        # the number of dense block
        self.dense_block = dense_block
        # growth rate
        self.growth_rate = growth_rate
        # bottleneck size
        self.bottleneck_size = bottleneck_size

        # using subpixel or deconv
        self.is_subpixel = is_subpixel
        # whether to use batch normalization
        self.is_bn = is_bn

        self.learning_rate = 0.0001
        self.global_step = tf.Variable(0,dtype=tf.int64,trainable=False,name="global_step")
        self.data_dict = loadWeightsData('pretrained/vgg16.npy')
        self.lamba = 0.001

    def buildModel(self):
        print("Building DenseNet...")

        # input layer
        x = tl.InputLayer(self.input, name='inputlayer')

        '''
        extract low level feature
        In Paper <Densely Connected Convolutional Networks>,the filter size here is 7*7
        and followed by a max pool layer
        upscale_input = tl.Conv2d(x,self.feature_size, [7, 7], act = None, name = 'conv0')
        upscale_input = tl.MaxPool2d(upscale_input, [3,3], [2,2], name = 'maxpool0')
        '''
        upscale_input = tl.Conv2d(x,self.feature_size, [3, 3], act = None, name = 'conv0')

        # dense-net
        '''
        using SRDenseNet_All model :
        all levels of features(output of dense block) are combined 
        via skip connections as input for reconstructing the HR images
        x
        |\
        | \
        |  dense blockl layer
        | /
        |/
        x1
        |
        [x,x1] (concat)
        '''
        x = upscale_input
        for i in range(self.dense_block):
            # the output of dense blocl
            x = self.__denseBlock(x, self.growth_rate, self.num_layers, [3,3] , layer = i)
            # concat
            upscale_input = tl.ConcatLayer([upscale_input,x],concat_dim=3,name='denseblock%d/concat_output'%(i))

        '''
        bottleneck layer
        In Paper <Image Super-Resolution Using Dense Skip Connections>
        The channel here is 256
        '''
        upscale_input = tl.Conv2d(upscale_input, self.bottleneck_size, [1,1], act=None, name = 'bottleneck')

        '''
        Paper <Densely Connected Convolutional Networks> using deconv layer to upscale the output
        here provide two methods here: deconv, subpixel
        '''
        # subpixel to upscale
        if self.is_subpixel:
            upscale_output = tl.Conv2d(upscale_input, self.bottleneck_size, [3, 3], act = None, name = 's1/1')
            upscale_output = tl.SubpixelConv2d(upscale_output, scale = 2, act=tf.nn.relu, name='pixelshufferx2/1')

            if self.scale == 4:
                upscale_output = tl.Conv2d(upscale_output, self.output_channels * 2 * 2, [3, 3], act = None, name = 's1/2')
                upscale_output = tl.SubpixelConv2d(upscale_output, scale = 2, act=tf.nn.relu, name='pixelshufferx2/2')
            elif self.scale == 8:
                upscale_output = tl.Conv2d(upscale_output, self.output_channels * 2 * 2, [3, 3], act=None, name='s1/2')
                upscale_output = tl.SubpixelConv2d(upscale_output, scale=2, act=tf.nn.relu,name='pixelshufferx2/2')

                upscale_output = tl.Conv2d(upscale_output, self.output_channels * 2 * 2, [3, 3], act=None, name='s1/3')
                upscale_output = tl.SubpixelConv2d(upscale_output, scale=2, act=tf.nn.relu,name='pixelshufferx2/3')

        # deconv to upscale
        else:
            # if scale is 8,using 3 deconv layers
            # is scale is 4,using 2 deconv layers
            width, height = int(upscale_input.outputs.shape[1]), int(upscale_input.outputs.shape[2])
            upscale_output, feature_size, width, height = self.__deconv(upscale_input, self.bottleneck_size, width, height, name='deconv0')
            upscale_output, feature_size, width, height = self.__deconv(upscale_output, feature_size, width, height,name='deconv1')
            if self.scale == 8:
                upscale_output, feature_size, width, height = self.__deconv(upscale_output, feature_size, width, height,name='deconv2')

        # reconstruction layer
        # output = tl.Conv2d(upscale_output, self.output_channels, [3, 3], act=tf.nn.relu, name='lastLayer')
        output = upscale_output
        output = tf.clip_by_value(output.outputs, 0.0, 1.0)
        self.output = output

        self.cacuDenseNetLoss(output)

        # Tensorflow graph setup... session, saver, etc.
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        print("Done building!")


    '''
    the implementation of dense block
    a denseblock is defined in the paper as
        x
        |\
        | \
        |  BN
        |  relu
        |  conv2d
        | /
        |/
        x1
        |
        [x,x1](concat)
        
    for a dense block which has n layers,the output is [x,x1,x2....xn]
    while xi mean the output of i-th layers in this dense block

    x: input to pass through the denseblock
    '''
    def __denseBlock(self , x, growth_rate = 16, num_layers = 8, kernel_size = [3, 3],layer = 0):
        dense_block_output = x
        for i in range(num_layers):
            '''
            In Paper <Densely Connected Convolutional Networks>
            each composite function contains three consecutive operations:
            batch normalization(BN), followed by a rectified linear unit (ReLU) and a 3*3 convolution (Conv).
            '''
            if self.is_bn:
                x = tl.BatchNormLayer(x,name = 'denseblock%d/BN%d'%(layer,i))
            x = ReluLayer(x,name = 'denseblock%d/relu%d'%(layer,i))
            x = tl.Conv2d(x,growth_rate,kernel_size,name = 'denseblock%d/conv%d'%(layer,i))
            # concat the output of layer
            dense_block_output = tl.ConcatLayer([dense_block_output,x],concat_dim=3,name = 'denseblock%d/concat%d'%(layer,i))
            x = dense_block_output

        return dense_block_output


    '''
    devonc layer
    for the input shape is  n * width * height * feature_size
    the output shape of the deconv layers is n * (width * 2) * (height * 2) * (feature_size / 2)
    '''
    def __deconv(self,x,feature_size,width,height,name='deconv2'):
        feature_size = feature_size // 2
        width, height = width * 2,height * 2
        # deconv layer
        deconv_output = tl.DeConv2d(x,feature_size,[3,3],[width,height],act=tf.nn.relu,name=name)
        return deconv_output,feature_size,width,height


    def cacuDenseNetLoss(self,output):
        l1_loss = tf.reduce_mean(tf.losses.absolute_difference(self.target, output))
        l2_loss = tf.reduce_mean(tf.squared_difference(self.target, output))

        vgg_target = custom_Vgg16(self.target,data_dict=self.data_dict)
        feature_target = [vgg_target.conv1_2, vgg_target.conv2_2, vgg_target.conv3_3, vgg_target.conv4_3, vgg_target.conv5_3]
        vgg_output = custom_Vgg16(output,data_dict=self.data_dict)
        feature_output = [vgg_output.conv1_2, vgg_output.conv2_2, vgg_output.conv3_3, vgg_output.conv4_3, vgg_output.conv5_3]


        per_loss_list = []
        for f,f_ in zip(feature_target,feature_output):
            per_loss_list.append(tf.reduce_mean(tf.squared_difference(f,f_)))

        per_loss = self.lamba * tf.reduce_sum(per_loss_list)

        loss = l1_loss + l2_loss + per_loss

        learning_rate = utils.learning_rate_decay(self.learning_rate,self.global_step)
        #Using adam optimizer as mentioned in the paper
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #This is the train operation for our objective
        self.train_op = optimizer.minimize(loss,global_step=self.global_step)
        # self.train_op = optimizer.minimize(loss,global_step=self.global_step)

        PSNR = utils.psnr_tf(self.target, output)

        # Scalar to keep track for loss
        summary_l1_loss = tf.summary.scalar("l1-loss", l1_loss)
        summary_l2_loss = tf.summary.scalar("l2-loss", l1_loss)
        summary_per_loss = tf.summary.scalar("per-loss", per_loss)
        summary_loss = tf.summary.scalar("loss", loss)
        summary_psnr = tf.summary.scalar("PSNR", PSNR)

        streaming_l1_loss, self.streaming_l1_loss_update = tf.contrib.metrics.streaming_mean(l1_loss)
        streaming_l1_loss_scalar = tf.summary.scalar('l1-loss',streaming_l1_loss)

        streaming_l2_loss, self.streaming_l2_loss_update = tf.contrib.metrics.streaming_mean(l2_loss)
        streaming_l2_loss_scalar = tf.summary.scalar('l2-loss',streaming_l2_loss)

        streaming_per_loss, self.streaming_per_loss_update = tf.contrib.metrics.streaming_mean(per_loss)
        streaming_per_loss_scalar = tf.summary.scalar('per-loss',streaming_per_loss)

        streaming_loss, self.streaming_loss_update = tf.contrib.metrics.streaming_mean(loss)
        streaming_loss_scalar = tf.summary.scalar('loss',streaming_loss)

        streaming_psnr, self.streaming_psnr_update = tf.contrib.metrics.streaming_mean(PSNR)
        streaming_psnr_scalar = tf.summary.scalar('PSNR',streaming_psnr)

        # Image summaries for input, target, and output
        '''
        input_image = tf.summary.image("input_image", tf.cast(self.input, tf.uint8))
        target_image = tf.summary.image("target_image", tf.cast(self.target, tf.uint8))
        output_image = tf.summary.image("output_image", tf.cast(output.outputs, tf.uint8))
        '''

        self.train_merge = tf.summary.merge([summary_l1_loss,summary_l2_loss,summary_psnr,summary_per_loss,summary_loss])
        self.test_merge = tf.summary.merge([streaming_l1_loss_scalar,streaming_psnr_scalar,streaming_per_loss_scalar,streaming_loss_scalar,streaming_l2_loss_scalar])


    def gram_matrix(x):
        assert isinstance(x, tf.Tensor)
        b, h, w, ch = x.get_shape().as_list()
        features = tf.reshape(x, [b, h * w, ch])
        # gram = tf.batch_matmul(features, features, adj_x=True)/tf.constant(ch*w*h, tf.float32)
        gram = tf.matmul(features, features, adjoint_a=True) / tf.constant(ch * w * h, tf.float32)
        return gram