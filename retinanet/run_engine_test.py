# coding:utf-8
"""
@Author  : b1xian
@Time    : 2019/12/18
"""
import torch
import retinanet
from retinanet._C import Engine
import time

import os

import cv2
import numpy as np


abnormal_engine_path="/home/bixian/work_space/models/zjrq/retinanet_rn18fpn_owl70000_finetune_90000_fp16.plan"
# abnormal_engine_path="/home/bixian/work_space/models/zjrq/retinanet_rn18fpn_owl70000_finetune_90000_fp16.plan"

ABNORMAL_INPUT_SIZE = (1280, 800)
model = Engine.load(abnormal_engine_path)
global resize_image


def gamma_trans(img, gamma=0.6):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def abnormal_preprocess(image):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    # print('origin image: ', image.shape, image)
    image = cv2.resize(image, ABNORMAL_INPUT_SIZE)
    image = gamma_trans(image, 0.6)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    resize_image = np.asarray(image)
    cv2.imwrite("./resize.jpg", image)
    image = image/255.
    # print(image)
    image = image - np.expand_dims(np.expand_dims(means, 0), 0)
    image = image / np.expand_dims(np.expand_dims(stds, 0), 0)
    image = image.transpose((2, 0, 1))
    # print('image shape: ', image.shape)
    return image, resize_image


def torchTest():
    global resize_image
    start = time.time()
    image = np.asarray(img)
    image = abnormal_preprocess(image)
    batch_image = []
    batch_image.append(image)
    # print(batch_image)
    data = torch.FloatTensor(batch_image)
    infer_data = data.cuda()
    # print(infer_data)
    # start = time.time()
    # for i in range(0, 100):
    #     scores, boxes, cls = Engine.infer(model, infer_data)
    # cost = time.time() - start
    # print('mean infer cost:', cost / 100)

    scores, boxes, cls = Engine.infer(model, infer_data)
    scores, boxes, cls = scores.cpu(), boxes.cpu(), cls.cpu()
    scores = np.array(scores)[0]
    boxes = np.array(boxes)[0]
    cls = np.array(cls)[0]
    print(time.time() - start)
    # print(scores, boxes, cls)
    # print(np.size(scores))
    # print(np.size(boxes, 0), np.size(boxes, 1))
    for i in range(0, np.size(boxes, 0)):
        # print(i, "scores:", scores[i], "cls:", cls[i], "boxes:[", boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], "]")
        if scores[i] > 0.6:
            cv2.rectangle(resize_image, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 0), 2)
            #cv2.imwrite("./result" + str(i) + ".jpg", resize_image)
    cv2.imwrite("./result.jpg", resize_image)


def numpyTest(image):
    print(dir(Engine))
    abnormal_engine = Engine.load(abnormal_engine_path)
    abnormal_batch_size = 1
    abnormal_max_detections = 50
    start = time.time()
    image,resize_image = abnormal_preprocess(image)
    print('process:', time.time() - start)
    # print(image)
    abnormal_input = []
    abnormal_input.append(image)
    abnormal_input = np.asarray(abnormal_input)
    # print(abnormal_input.shape)
    abnormal_input = abnormal_input.flatten()
    start_1 = time.time()
    # for i in range(0, 100):
    #     scores, boxes, cls = Engine.engine_infer(abnormal_engine, abnormal_input)
    scores, boxes, cls = Engine.engine_infer(abnormal_engine, abnormal_input)
    print(scores.shape)
    print(boxes.shape)
    print(cls.shape)
    infer_cost = time.time()-start_1
    print('infer cost:', infer_cost)
    # print('mean infer cost:', infer_cost / 100)
    print('total cost',time.time() - start)
    # keep = scores > 0.5
    # scores = scores[keep]
    # boxes = boxes[keep]
    # cls = cls[keep]
    start = time.time()
    batch_targets = []

    # boxes = np.reshape(boxes, (-1, 4))
    # for bs in range(0, abnormal_batch_size):
    #     frame_targets = []
    #     batch_scores = scores[bs*abnormal_max_detections:(bs+1)*abnormal_max_detections]
    #     keep = batch_scores > 0.5
    #     batch_scores = batch_scores[keep]
    #     print(batch_scores.shape)
    #     if len(batch_scores) > 0:
    #         batch_boxes = boxes[bs*abnormal_max_detections:(bs+1)*abnormal_max_detections*4]
    #         batch_boxes = batch_boxes[keep]
    #         batch_cls = cls[bs*abnormal_max_detections:(bs+1)*abnormal_max_detections]
    #         batch_cls = batch_cls[keep]
    #
    #         for i, score in enumerate(batch_scores):
    #             frame_targets.append({'score': score,
    #                                   'cls': int(batch_cls[i]),
    #                                   'box': batch_boxes[i]})
    #     batch_targets.append(frame_targets)

    for bs in range(0, abnormal_batch_size):
        frame_targets = []
        for i in range((bs*abnormal_max_detections), ((bs+1)*abnormal_max_detections)):
            # print(i, "scores:", scores[i], "cls:", cls[i], "boxes:[", boxes[i*4+0], boxes[i*4+1], boxes[i*4+2], boxes[i*4+3], "]")
            if scores[i] > 0.5:
                frame_targets.append({'score': scores[i],
                                      'cls': cls[i],
                                      'box': [boxes[i*4+0], boxes[i*4+1], boxes[i*4+2], boxes[i*4+3]]})
                cv2.rectangle(resize_image, (boxes[i*4+0], boxes[i*4+1]), (boxes[i*4+2], boxes[i*4+3]), (0, 255, 0), 1)
        batch_targets.append(frame_targets)
    print('get targets cost:', time.time()-start)
    print(batch_targets)
    cv2.imwrite("./result_fp32.jpg", resize_image)


def video_test(video_path, out_path):
    writeVideo_flag = True
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_capture = cv2.VideoCapture(video_path)

    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    print('>>>>>>>>>>>>>>>>>>>>>', width, height)
    abnormal_engine = Engine.load(abnormal_engine_path)

    fps = 12
    out = cv2.VideoWriter(out_path, fourcc, fps, (1280, 800), True)
    # 赣瑞龙线k104+550公跨铁赣向
    # alarm_range = [[[5, 282], [472,187], [700, 204], [464, 571], [5,571]]]
    # dh_160
    # alarm_range = [[[0, 670], [731, 185], [1443, 185], [1881, 1080], [0, 1080]]]
    # alarm_range = [[[625, 184], [624, 239], [591, 325], [465, 410], [11, 621], [10, 1062], [1905, 1064], [1905, 723], [1311, 284],
    #   [1123, 211], [977, 172]]]
    # alarm_range = [[[0,803],[611,326],[1069,326],[1920,800],[1920,1080],[0,1080]]]
    # alarm_range = [[[0,672], [633,330],[631,204],[1131,190],[1386,311],[1920,677],[1920,1080],[0,1080]]]
    # hk_159
    # alarm_range = [[[0, 775], [724, 302], [1377, 298], [1920, 1080], [0, 1080]]]
    # alarm_range = [[[1027, 122], [931, 198], [630, 342], [93, 598], [147, 1071], [1909, 1072], [1909, 922], [1439, 210], [1285, 110]]]
    # alarm_range = [[[0, 566], [725, 139], [1062, 139], [1774, 1080], [0, 1080]]]

    # hk_161
    # alarm_range = [[[0, 925], [1119, 337], [1645, 353], [1913, 585], [1920, 1080], [0, 1080]]]
    # hk_162
    # alarm_range = [[[0, 773], [511, 405], [1511, 405], [1920, 603], [1920, 1080], [0, 1080]]]
    # 4k2
    # alarm_range = [[[855, 563], [1059, 569], [1331, 627], [1809, 1007], [27, 951], [891, 623]]]

    # 衢九线k223+908小岗村隧道铁塔线路监控九向
    # alarm_range = [[[694, 150], [893, 68], [1910, 639], [1881, 1048]]]
    # 衢九线
    # alarm_range = [[[0, 743], [1221, 421], [1533, 427], [1651, 1075], [0, 1075]]]
    # 赣瑞龙线k104+550公跨铁龙向
    # alarm_range = [[[70, 184], [417, 172], [690, 248], [695, 560], [305, 554]]]
    # 武九线K157
    # alarm_range = [[[343, 1080], [1077, 33], [1387, 45], [1063, 1080]]]
    # hk_157
    # alarm_range = [[[0, 510], [524, 222], [955, 225], [1131, 720], [0, 720]]]
    # 4k2
    alarm_range = [[[855, 563], [1059, 569], [1331, 627], [1809, 1007], [27, 951], [891, 623]]]
    # 4k8
    # alarm_range = [[[687, 295], [887, 319], [987, 433], [1199, 1047], [3, 1035], [711, 409]]]
    # 20160809_004200_20160809_011000
    # alarm_range = [[[589, 350], [536, 358], [506, 408], [464, 716], [1249, 706], [1250, 507], [828, 394]]]
    # dh_158
    # alarm_range = [[[0, 640], [739, 315], [1317, 315], [1900, 1080], [0, 1080]]]
    # 093
    # alarm_range = [[[255, 1080], [1117, 455], [1270, 455], [1233, 1080]]]
    start_time = time.time()
    frame_count = 0
    total_cost = 0.
    while True:
        video_capture.grab()
        ret, frame = video_capture.read()
        if ret != True:
            print('break')
            break
        frame_count += 1
        camera_time = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        # print('------>camera time', camera_time)
        #         if camera_time > 48:
        #             break
        abnormal_batch_size = 1
        abnormal_max_detections = 100
        start = time.time()
        image, resize_image = abnormal_preprocess(frame)
        # resize_image = gamma_trans(resize_image, 0.6)
        print('process:', time.time() - start)
        # print(image)
        abnormal_input = []
        abnormal_input.append(image)
        abnormal_input = np.asarray(abnormal_input)
        # print(abnormal_input.shape)
        abnormal_input = abnormal_input.flatten()
        start_time_1 = time.time()
        scores, boxes, cls = Engine.engine_infer(abnormal_engine, abnormal_input)
        infer_cost = time.time()-start_time_1
        print('infer cost:', infer_cost)
        # print('mean infer cost:', infer_cost / 100)
        # print(scores)
        print('total cost',time.time() - start)
        # for bs in range(0, abnormal_batch_size):
        #     for i in range((bs*abnormal_max_detections), ((bs+1)*abnormal_max_detections)):
        #         if scores[i] > 0.5:
        #             cv2.rectangle(resize_image, (boxes[i*4+0], boxes[i*4+1]), (boxes[i*4+2], boxes[i*4+3]), (0, 255, 0), 1)
        #             cv2.putText(resize_image, '%.2f' % scores[i], (boxes[i*4+0], boxes[i*4+1]), 0, 0.5, (6, 254, 15), 2)
        # out.write(resize_image)
        # cv2.imshow('out_frame', resize_image)
        if cv2.waitKey(1):
            continue

    out.release()
    video_capture.release()
    end_time = time.time()
    print(end_time - start_time)


if __name__ == '__main__':
    # torchTest()
    numpyTest(cv2.imread('test.jpg'))
    # pre_path = '/media/bixian/新加卷1/dataset/zjrq/'
    # path_name = 'test_video'
    # # video_name = '4K2_snow_3'
    # out_path_name = 'test_video_TRT_16_result'
    # if not os.path.exists(os.path.join(pre_path,out_path_name)):
    #     os.makedirs(os.path.join(pre_path,out_path_name))
    # for f in os.listdir(os.path.join(pre_path,path_name)):
    #     video_name = f[:f.index('.')]
    #     print(f'detect video{video_name}...')
    #     video_path = os.path.join(pre_path, path_name, video_name+'.mp4')
    #
    #     out_path = os.path.join(pre_path, out_path_name, video_name+'_fp16_detect.mp4')
    #     video_test(video_path, out_path)

