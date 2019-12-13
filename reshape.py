# coding:utf-8
"""
@Author  : b1xian
@Time    : 2019/12/5
"""
import torch
import cv2
from PIL import Image
import time
import numpy
from torchvision import transforms

if __name__ == '__main__':
    '''torch.Size([749, 1333, 3])
torch.Size([3, 749, 1333])
    '''
    resize  = 800
    max_size = 1333
    image = cv2.imread('/media/bixian/新加卷1/dataset/zjrq/coconight/images/val/4K2_20191112_202521_094_120.jpg')
    h, w,c = image.shape
    # image_ = image.reshape(c, h, w)
    # print(image_.shape)
    # h, w = image.shape[0:2]
    # print(image.shape)
    # ratio = resize / min(w, h)
    # if ratio * max(w, h) > max_size:
    #     ratio = max_size / max(w, h)
    # im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # print(im.size)
    # im = im.resize((int(ratio * d) for d in im.size), Image.BILINEAR)
    # print(im.size)
    # print(*im.size[::-1])
    # print( len(im.mode))
    # image = image.reshape((1, image.shape[0], image.shape[1],
    # image.shape[2]))
    # print(image.shape)
    #
    # tensor3 = torch.Tensor(image)

    ratio = resize / min(w, h)
    if ratio * max(w, h) > max_size:
        ratio = max_size / max(w, h)
    # print('ratio:', ratio)
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (int(w * ratio), int(h * ratio)))
    # print(im.shape)
    im = im / 255
    # im_reshape = im.reshape(im.shape[2], im.shape[0], im.shape[1])
    # data = torch.Tensor(im_reshape).float()
    data = torch.Tensor(im).float().permute(2, 0, 1)
    print(data.shape)

    start_1 = time.time()
    print('numpy cost:', time.time() - start_1)

    tensor3 = torch.Tensor(image)
    start_1 = time.time()
    tensor3 = tensor3.float().div(255).permute(2, 0, 1)
    print('torch cost:', time.time() - start_1)
    print(tensor3.shape)


