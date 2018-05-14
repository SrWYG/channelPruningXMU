from xmuutil import utils
from xmudata.scdata import SeparateChannelData

class WaveletData(object):
    def __init__(self, data):
        self.data = SeparateChannelData(data)

    def get_test_set(self,batch_size):
        x_imgs,y_imgs = self.data.get_test_set(batch_size)
        x_dwt_imgs = utils.get_dwt_images(x_imgs)
        y_dwt_imgs = utils.get_dwt_images(y_imgs)
        return x_dwt_imgs,y_dwt_imgs


    def get_batch(self, batch_size):
        x_imgs, y_imgs = self.data.get_batch(batch_size)
        x_dwt_imgs = utils.get_dwt_images(x_imgs)
        y_dwt_imgs = utils.get_dwt_images(y_imgs)
        return x_dwt_imgs,y_dwt_imgs




