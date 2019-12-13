# coding:utf-8
"""
@Author  : b1xian
@Time    : 2019/12/7
"""
import os
from retinanet import infer_video
from retinanet.model import Model
import cv2
import torch
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
import numpy as np


class Retinanet:

    def __init__(self, model_path='/workspace/retinanet_rn18fpn_night_16000.pth', mixed_precision=True):
        self.load_model(model_path)
        self.mixed_precision = mixed_precision
        self.prepare_model()

    def load_model(self, model_path, verbose=False):
        model, state = Model.load(model_path)

        state['path'] = model_path
        self.model = model

    def prepare_model(self):

        backend = 'pytorch' if isinstance(self.model, Model) or isinstance(self.model, DDP) else 'tensorrt'
        if backend is 'pytorch':
            if torch.cuda.is_available(): model = self.model.cuda()
            model = amp.initialize(model, None,
                                   opt_level='O2' if self.mixed_precision else 'O0',
                                   keep_batchnorm_fp32=True,
                                   verbosity=0)
            model.eval()

    def gamma_trans(self, img, gamma=0.6):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)

    def predict(self, frame, alarm_range, camera_time):
        frame = self.gamma_trans(frame)
        return infer_video.detect(self.model, frame, alarm_range, camera_time)


if __name__ == '__main__':
    # img_path = '/workspace/test_imgs/night_hk_157_1720.jpg'
    # alarm_range = [[[21, 1073], [677, 313], [956, 361], [1296, 1066]]]
    # net = Retinanet()
    # img = cv2.imread(img_path)
    # alarm, frame = net.predict(img, alarm_range, 1)
    # cv2.imwrite('detect.jpg', frame)
    # print(alarm)


    alarm_range = [[[21, 1073], [677, 313], [956, 361], [1296, 1066]]]
    video_path = '/workspace/4K2_20191112_202521_094.mp4'

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_capture = cv2.VideoCapture(video_path)
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    out = cv2.VideoWriter('./4K2_20191112_202521_094_output.mp4', fourcc, 25, (w, h), True)

    net = Retinanet()

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_count += 1
        alarm, frame = net.predict(frame, alarm_range, frame_count)
        if alarm:
            print(alarm)
        out.write(frame)
        if cv2.waitKey(5):
            continue

    out.release()
    video_capture.release()