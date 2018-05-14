from xmudata.data import Data
import tensorlayer as tl
from xmuutil import utils
import os
from xmuutil.exception import LargeSizeException


class DIV2K2018(Data):

    def __init__(self, train_truth_dir, train_data_dir, test_truth_dir = None, test_data_dir=None, image_size = 96, scale = 4, train_postfix_len = 3, test_postfix_len = -1, test_per=0.01):
        Data.__init__(self, train_truth_dir, train_data_dir,test_truth_dir,test_data_dir, train_postfix_len, test_postfix_len, test_per)
        self.image_size = image_size
        self.scale = scale

    def get_image_set(self, image_lr_list,input_dir,ground_truth_dir, postfix_len):
        y_imgs = []
        x_imgs = []
        # use 10 threads to read files
        imgs_lr = tl.visualize.read_images(image_lr_list, input_dir)
        image_hr_list = utils.get_hrimg_list(image_lr_list, postfix_len)
        imgs_hr = tl.visualize.read_images(image_hr_list, ground_truth_dir)

        for i in range(len(imgs_lr)):
            #crop the image randomly
            try:
                x_img,y_img = utils.crop(imgs_lr[i], imgs_hr[i], self.image_size, self.image_size, self.scale, is_random=True)
            except LargeSizeException as e:
                print(e)
            else:
                y_imgs.append(y_img)
                x_imgs.append(x_img)
        return x_imgs, y_imgs




