import torch
import torch.nn.functional as F
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
import numpy as np
import cv2
from PIL import Image
import time
from .model import Model

def iou(bbox1, bbox2):
    """Compute the iou of two boxes.
    Parameters
    ----------
    bbox1, bbox2: list.
        The bounding box coordinates: [xmin, ymin, xmax, ymax]
    Returns
    -------
    iou: float.
        The iou of bbox1 and bbox2.
    ior: float.
        inner area of bbox2.
    """
    xmin1, ymin1 = int(bbox1[0] - bbox1[2] / 2.0), int(bbox1[1] - bbox1[3] / 2.0)
    xmax1, ymax1 = int(bbox1[0] + bbox1[2] / 2.0), int(bbox1[1] + bbox1[3] / 2.0)
    xmin2, ymin2 = int(bbox2[0] - bbox2[2] / 2.0), int(bbox2[1] - bbox2[3] / 2.0)
    xmax2, ymax2 = int(bbox2[0] + bbox2[2] / 2.0), int(bbox2[1] + bbox2[3] / 2.0)

    # 获取矩形框交集对应的顶点坐标(intersection)
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    # 计算交集面积
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    # 计算交并比
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou, inter_area / area2

def invade_check(frame, img_size, invade_pts, target_box):
    '''
    判断目标框是否侵入预警区域
    :param invade_pts:
    :param target_box: [x, y, x+w, y+h, w, h]
    :return:
    '''
    target_mask = np.zeros((img_size[1], img_size[0]), dtype="uint8")
    invade_mask = np.zeros((img_size[1], img_size[0]), dtype="uint8")
    cv2.fillPoly(invade_mask, invade_pts, 255)
    target_pts = np.array([[[target_box[0], target_box[1]],
                           [target_box[0]+target_box[4], target_box[1]],
                           [target_box[0]+target_box[4], target_box[1]+target_box[5]],
                           [target_box[0], target_box[1]+target_box[5]]]])
    cv2.fillPoly(target_mask, target_pts, 255)
    # cv2.fillPoly(frame, invade_pts, (255, 0, 0))
    # cv2.fillPoly(frame, target_pts, (0, 0, 255))
    masked_and = cv2.bitwise_and(invade_mask, target_mask, mask=invade_mask)
    masked_or = cv2.bitwise_or(invade_mask, target_mask)
    or_area = np.sum(np.float32(np.greater(masked_or, 0)))
    and_area = np.sum(np.float32(np.greater(masked_and, 0)))
    IOU = and_area / or_area
    return IOU > 0


def infer(model, path, alarm_range, resize=800, max_size=1333, mixed_precision=True, verbose=True):
    alarm_range = np.array(alarm_range, np.int32)
    backend = 'pytorch' if isinstance(model, Model) or isinstance(model, DDP) else 'tensorrt'

    # Prepare model
    if backend is 'pytorch':
        if torch.cuda.is_available(): model = model.cuda()
        model = amp.initialize(model, None,
                           opt_level = 'O2' if mixed_precision else 'O0',
                           keep_batchnorm_fp32 = True,
                           verbosity = 0)
        model.eval()

    if verbose:
        print('   backend: {}'.format(backend))
        print('     batch: {}, precision: {}'.format(1,
            'unknown' if backend is 'tensorrt' else 'mixed' if mixed_precision else 'full'))
        print('Running inference...')

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_capture = cv2.VideoCapture(path)
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    out = cv2.VideoWriter('./output.mp4', fourcc, 25, (w, h), True)

    stride = model.module.stride if isinstance(model, DDP) else model.stride
    start = time.time()
    frame_count = 0
    with torch.no_grad():
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_count += 1
            start_1 = time.time()
            ratio = resize / min(w, h)
            if ratio * max(w, h) > max_size:
                ratio = max_size / max(w, h)
            # print('ratio:', ratio)
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (int(w*ratio), int(h*ratio)))
            print('preprocess0 cost:', time.time() - start_1)
            # rgb
            # im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
            # im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
            # im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225
            start_1 = time.time()
            # im_reshape = im.reshape(im.shape[2], im.shape[0], im.shape[1])
            # data = torch.Tensor(im_reshape).float()
            data = torch.Tensor(im).float().div(255).permute(2, 0, 1)
            # data = data.float().div(255).view(im.shape)
            # print('preprocess0 cost:', time.time() - start_1)
            # im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # ratio = resize / min(w, h)
            # if ratio * max(w, h) > max_size:
            #     ratio = max_size / max(w, h)
            # im = im.resize((int(ratio * d) for d in im.size), Image.BILINEAR)
            # data = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
            # print(im.size[::-1], len(im.mode))
            # data = data.float().div(255).view(*im.size[::-1], len(im.mode))

            # print('preprocess1 cost:', time.time() - start_1)
            start_1 = time.time()
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            for t, mean, std in zip(data, mean, std):
                t.sub_(mean).div_(std)

            # print('preprocess2 cost:', time.time() - start_1)
            # Apply padding
            pw, ph = ((stride - d % stride) % stride for d in [im.shape[1], im.shape[0]])
            # # print(pw,ph)
            start_1 = time.time()
            # print(data.shape)
            data = F.pad(data, (0, pw, 0, ph))
            # print(data.shape)
            # print('preprocess3 cost:', time.time() - start_1)
            start_1 = time.time()
            data = torch.unsqueeze(data, 0)
            data = data.cuda(non_blocking=True)
            # print('preprocess4 cost:', time.time() - start_1)

            start_1 = time.time()
            scores, boxes, classes = model(data)
            # print('inference cost:', time.time() - start_1)
            scores = scores.cpu().view(-1)
            boxes = boxes.cpu().view(-1, 4)
            classes = classes.cpu().view(-1)

            keep = (scores > .5).nonzero()
            scores = scores[keep]
            boxes = boxes[keep, :] / ratio
            classes = classes[keep].int()
            alarm = False
            for score, box, cat in zip(scores, boxes, classes):
                x1, y1, x2, y2 = box.data.tolist()[0]
                cat = cat.item()
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                # 对检测出的目标框做是否入侵判断
                if invade_check(frame, (frame.shape[1], frame.shape[0]), alarm_range, [x1, y1, x2, y2, x2-x1, y2-y1]):
                    alarm = frame_count
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.polylines(frame, [alarm_range.reshape((-1, 1, 2))], True, (0, 255, 255), 2)
            out.write(frame)
            print(alarm)
    print('time cost:', time.time() - start)
    print('mean fps:', frame_count / int(time.time() - start))
    out.release()
    video_capture.release()


def detect(model, frame, alarm_range, camera_time, resize=800, max_size=1333, mixed_precision=True, verbose=False):
    alarm_range = np.array(alarm_range, np.int32)
    backend = 'pytorch' if isinstance(model, Model) or isinstance(model, DDP) else 'tensorrt'

    # Prepare model
    # if backend is 'pytorch':
    #     if torch.cuda.is_available(): model = model.cuda()
    #     model = amp.initialize(model, None,
    #                            opt_level='O2' if mixed_precision else 'O0',
    #                            keep_batchnorm_fp32=True,
    #                            verbosity=0)
    #     model.eval()

    if verbose:
        print('   backend: {}'.format(backend))
        print('     batch: {}, precision: {}'.format(1,
                                                     'unknown' if backend is 'tensorrt' else 'mixed' if mixed_precision else 'full'))
        print('Running inference...')


    stride = model.module.stride if isinstance(model, DDP) else model.stride
    with torch.no_grad():
        h,w,c = frame.shape
        ratio = resize / min(w, h)
        if ratio * max(w, h) > max_size:
            ratio = max_size / max(w, h)
        # print('ratio:', ratio)
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (int(w * ratio), int(h * ratio)))
        data = torch.Tensor(im).float().div(255).permute(2, 0, 1)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for t, mean, std in zip(data, mean, std):
            t.sub_(mean).div_(std)

        pw, ph = ((stride - d % stride) % stride for d in [im.shape[1], im.shape[0]])
        # # print(pw,ph)
        start_1 = time.time()
        # print(data.shape)
        data = F.pad(data, (0, pw, 0, ph))
        # print(data.shape)
        # print('preprocess3 cost:', time.time() - start_1)
        start_1 = time.time()
        data = torch.unsqueeze(data, 0)
        data = data.cuda(non_blocking=True)
        # print('preprocess4 cost:', time.time() - start_1)

        start_1 = time.time()
        scores, boxes, classes = model(data)
        # print('inference cost:', time.time() - start_1)
        scores = scores.cpu().view(-1)
        boxes = boxes.cpu().view(-1, 4)
        classes = classes.cpu().view(-1)

        keep = (scores > .5).nonzero()
        scores = scores[keep]
        boxes = boxes[keep, :] / ratio
        classes = classes[keep].int()
        alarm = False
        for score, box, cat in zip(scores, boxes, classes):
            x1, y1, x2, y2 = box.data.tolist()[0]
            cat = cat.item()
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            # 对检测出的目标框做是否入侵判断
            if invade_check(frame, (frame.shape[1], frame.shape[0]), alarm_range,
                            [x1, y1, x2, y2, x2 - x1, y2 - y1]):
                alarm = camera_time
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.polylines(frame, [alarm_range.reshape((-1, 1, 2))], True, (0, 255, 255), 2)

    return alarm, frame

