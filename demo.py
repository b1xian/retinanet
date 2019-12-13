# coding:utf-8
"""
@Author  : b1xian
@Time    : 2019/12/7
"""
import cv2
import os
import time
from Retinanet import Retinanet
import argparse


def detect_video(model, input_path, output_path, video_name, alarm_range):

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_capture = cv2.VideoCapture(os.path.join(input_path, video_name))
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    out = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, 25, (w, h), True)

    start = time.time()
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_count += 1
        alarm, frame = model.predict(frame, alarm_range, frame_count)
        if alarm:
            print(alarm)
        else:
            print(False)
        out.write(frame)
    cost = time.time() - start
    print('mean fps:', frame_count / (cost + 1e-5))
    out.release()
    video_capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="输入模型文件")
    parser.add_argument("--input_path", required=True, help="输入测试视频文件夹路径")
    parser.add_argument("--output_path", required=True, help="输出测试视频文件夹路径")
    args = parser.parse_args()
    model_path = args.model_path
    input_path = args.input_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = Retinanet(model_path=model_path)


    alarm_range = [[[21, 1073], [677, 313], [956, 361], [1296, 1066]]]
    for f in os.listdir(input_path):
        print('start detect video : ', f)
        detect_video(model, input_path, output_path, f, alarm_range)




