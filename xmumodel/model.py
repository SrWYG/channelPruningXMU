import tensorflow as tf
from tqdm import tqdm
from abc import ABCMeta,abstractmethod
import os
import shutil
from xmuutil import utils


"""
An implementation of the neural network used for
super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""
class Model(object, metaclass=ABCMeta):
    def __init__(self, num_layers=32, feature_size=256, scale=8, input_channels=3, output_channels=3, prunedlist = [0,0,0,0 ,0,0,0,0 ,0,0,0,0 ,0,0,0,0], prunesize=0):
        self.num_layers = num_layers
        self.feature_size = feature_size
        self.scale = scale
        self.output_channels = output_channels
        self.input_channels = input_channels
        #self.prunedsize = prunedsize
        self.prunesize = prunesize
        self.prunedlist = prunedlist
        #Placeholder for image inputs
        self.input = tf.placeholder(tf.float32, [None, None, None, input_channels], name='input')
        #Placeholder for upscaled image ground-truth
        self.target = tf.placeholder(tf.float32, [None, None, None, output_channels], name='output')
        self.output = None

        self.tran_op = None

        self.summaries = []
    

    @abstractmethod
    def buildModel(self):
        pass

    """
    Save the current state of the network to file
    """
    def save(self, savedir='saved_models', global_step=None):
        print("Saving...")
        # tl.files.save_npz(self.all_params, name=savedir + '/model.npz', sess=self.sess)
        self.saver.save(self.sess,savedir+"/model", global_step=global_step)
        print("Saved!")

    """
    Resume network from previously saved weights
    """
    def resume(self,savedir='saved_models',global_step=None):

        if os.path.exists(savedir):
            if global_step is None:
                checkpoint_path_to_resume = tf.train.latest_checkpoint(savedir)
            else:
                checkpoint_path_to_resume = None
                checkpoint_path_list = tf.train.get_checkpoint_state(savedir)
               # prefix_to_delete = checkpoint_path_list.model_checkpoint_path + '-'
                hyphen_pos = checkpoint_path_list.model_checkpoint_path.rfind('-')
                global_step_str = str(global_step)
                for checkpoint_path in checkpoint_path_list.all_model_checkpoint_paths:
                    checkpoint_path_iteration = checkpoint_path[hyphen_pos+1:]
                    if(checkpoint_path_iteration == global_step_str):
                        checkpoint_path_to_resume = checkpoint_path
                        break

                if checkpoint_path_to_resume is None:
                    checkpoint_path_to_resume = tf.train.latest_checkpoint(savedir)

            print("Restoring from " + checkpoint_path_to_resume)
            self.saver = tf.train.Saver() if self.saver == None else self.saver
            self.sess = tf.Session() if self.sess == None else self.sess
            self.saver.restore(self.sess,checkpoint_path_to_resume)
            print("Restored!")

    def cacuLoss(self, x):
        loss = tf.reduce_mean(tf.losses.absolute_difference(self.target, x.outputs))
        #Using adam optimizer as mentioned in the paper
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        #This is the train operation for our objective
        self.train_op = optimizer.minimize(loss)

        PSNR = utils.psnr_tf(self.target, x.outputs)

        # Scalar to keep track for loss
        summary_loss = tf.summary.scalar("loss", loss)
        summary_psnr = tf.summary.scalar("PSNR", PSNR)

        streaming_loss, self.streaming_loss_update = tf.contrib.metrics.streaming_mean(loss)
        streaming_loss_scalar = tf.summary.scalar('loss',streaming_loss)

        streaming_psnr, self.streaming_psnr_update = tf.contrib.metrics.streaming_mean(PSNR)
        streaming_psnr_scalar = tf.summary.scalar('PSNR',streaming_psnr)

        # Image summaries for input, target, and output
        input_image = tf.summary.image("input_image", tf.cast(self.input, tf.uint8))
        target_image = tf.summary.image("target_image", tf.cast(self.target, tf.uint8))
        output_image = tf.summary.image("output_image", tf.cast(x.outputs, tf.uint8))

        self.train_merge = tf.summary.merge([summary_loss,summary_psnr])
        self.test_merge = tf.summary.merge([streaming_loss_scalar,streaming_psnr_scalar])

    """
    Function to setup your input data pipeline
    """
    def set_data(self, data):
        self.data = data


    """
    Estimate the trained model
    x: (tf.float32, [batch_size, h, w, output_channels])
    """
    def predict(self, x):
        return self.sess.run(self.output, feed_dict={self.input: x})

    """
    Train the neural network
    """
    def train(self,batch_size= 10, iterations=1000,save_dir="saved_models",reuse=False,reuse_dir=None,reuse_epoch=None,log_dir="log"):

        #create the save directory if not exist
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.mkdir(log_dir)
        #Make new save directory
        os.mkdir(save_dir)
        #Operation to initialize all variables
        init = tf.global_variables_initializer()
        print("Begin training...")
        with self.sess as sess:
            #Initialize all variables
            sess.run(init)
            if reuse:
                self.resume(reuse_dir,reuse_epoch)
            #create summary writer for train
            train_writer = tf.summary.FileWriter(log_dir+"/train",sess.graph)

            #If we're using a test set, include another summary writer for that
            test_writer = tf.summary.FileWriter(log_dir+"/test",sess.graph)
            test_feed = []
            while True:
                test_x,test_y = self.data.get_test_set(batch_size)
                if test_x!=None and test_y!=None:
                    test_feed.append({
                            self.input:test_x,
                            self.target:test_y
                    })
                else:
                    break

            #This is our training loop
            for i in tqdm(range(iterations)):
                #Use the data function we were passed to get a batch every iteration
                x,y = self.data.get_batch(batch_size)
                #Create feed dictionary for the batch
                feed = {
                    self.input:x,
                    self.target:y
                }
                #Run the train op and calculate the train summary
                summary,_ = sess.run([self.train_merge,self.train_op],feed)
                #Write train summary for this step
                train_writer.add_summary(summary,i)
                #test every 10 iterations
                if i%200 == 0:
                    sess.run(tf.local_variables_initializer())
                    for j in range(len(test_feed)):
                        sess.run([self.streaming_loss_update,self.streaming_psnr_update],feed_dict=test_feed[j])
                    streaming_summ = sess.run(self.test_merge)
                    #Write test summary
                    test_writer.add_summary(streaming_summ,i)

                # Save our trained model
                if i!=0 and i % 500 == 0:
                    self.save(save_dir,i)

            self.save(save_dir)
            test_writer.close()
            train_writer.close()


