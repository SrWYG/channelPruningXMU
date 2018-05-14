import numpy as np
import os


class SampleData(object):
    # name_postfix for Track1: x8  Track2: x4m  Track3: x4d  Track4: x4w1(2/3/4)
    def __init__(self, data, sample_file=None, name_postfix='x8', repeat_times=4):
        self.data = data
        self.name_postfix = name_postfix
        self.repeat_times = repeat_times

        sample_hr_list = []
        f = open(sample_file, 'r')
        for line in f.readlines():
            hr_file = line.strip()
            sample_hr_list.append(hr_file)

        sample_lr_list = self.__get_lrimg_list(sample_hr_list)
        sample_lr_list = np.repeat(sample_lr_list, self.repeat_times)

        self.data.train_set.extend(sample_lr_list)

    def get_batch(self, batch_size):
        x_imgs, y_imgs = self.data.get_batch(batch_size)
        return x_imgs, y_imgs

    def get_test_set(self, batch_size):
        x_imgs, y_imgs = self.data.get_test_set(batch_size)
        return x_imgs,y_imgs

    def __get_lrimg_list(self,image_hr_list):
        image_lr_list= []
        for i in range(len(image_hr_list)):
            name_hr,postfix = os.path.splitext(image_hr_list[i])
            name_lr = name_hr + self.name_postfix + postfix
            image_lr_list.append(name_lr)
        return image_lr_list