# This nms is modified version of https://github.com/ultralytics/yolov5/blob/master/utils/general.py
# all torch dependeces is removed,only numpy is using for calculcations

import numpy as np


def xywh2xyxy(x):
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2

    return y


def _fast_nms(boxes, scores, iou_th):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes 
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)#[::-1]
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > iou_th)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick




# Like original NMS but working on NumPy
def non_max_supression(predictions, conf_thresh=0.25, iou_thresh=0.45, multilabel=False, agnostic=False, max_det=300):
    nc = predictions.shape[2] - 5  # number of classes
    xc = predictions[...,4] > conf_thresh # candidates  

    assert 0 <= conf_thresh <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thresh <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    min_wh, max_wh = 2, 4096 # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms() 

    multilabel &= nc > 1 # multiple labels per box (adds 0.5ms/img)


    output = [np.zeros((0, 6), dtype=predictions.dtype) for i in range(predictions.shape[0])]
    for xi, x in enumerate(predictions):
        x = x[xc[xi]]

        # If none remain process next image 
        if not x.shape[0]:
            continue

        # Compute conf: object conf * class conf
        x[:, 5:] *= x[:, 4:5]
       

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        

        # Detections matrix nx6 (xyxy, conf, cls)
        if multilabel:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else: # best class only
            conf = x[:, 5:].max(1, keepdims=True)
            j = x[:, 5:].argmax(1).astype(x.dtype)
            j = np.expand_dims(j, 1)
            
            x = np.concatenate((box, conf, j), 1)

        n = x.shape[0] # number of boxes 
        if not n:
            continue
        elif n > max_nms: # excess boxes 
            x = x[x[:, 4].argsort()][:max_nms] # sort by confidence

        
        bboxes = x[:, :4].copy() + x[:, 5:6].copy() * (0 if agnostic else max_wh)
        scores = x[:, 4]

        indexes = _fast_nms(bboxes, scores, iou_thresh)
        x = x[indexes]
        
        if x.shape[0] > max_det:
            x = x[x[:, 4].argsort()][:max_det]

        output[xi] = x


    return output
