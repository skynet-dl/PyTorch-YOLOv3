from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from timeit import default_timer as timer


def changeBGR2RGB(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    return img


def changeRGB2BGR(img):
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r

    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--vedio_file", type=str, default="../video/videoplayback3.mp4", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="./yolov3_ckpt_200d.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="config/custom.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    model.eval()  # Set in evaluation mode
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    #if opt.vedio_file.endswith(".mp4"):
    cap = cv2.VideoCapture(opt.vedio_file)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (640,360))

    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a=[]
    time_begin = time.time()
    NUM = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #NUM=0
    while cap.isOpened():
        ret, img = cap.read()
        if ret is False:
            break
        # img = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_CUBIC)

        RGBimg=changeBGR2RGB(img)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)

        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))

        start = timer()
        with torch.no_grad():
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        end = timer()
        print(f"exec time = {end - start}")

        a.clear()
        if detections is not None:
            a.extend(detections)
        b=len(a)
        if len(a)  :
            for detections in a:
                if detections is not None:
                    detections = rescale_boxes(detections, opt.img_size, RGBimg.shape[:2])
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        box_w = x2 - x1
                        box_h = y2 - y1
                        color = [int(c) for c in colors[int(cls_pred)]]
                        img = cv2.rectangle(img, (int(x1), int(y1 + box_h)), (int(x2), int(y1)), color, 2)
                        cv2.putText(img, classes[int(cls_pred)], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.putText(img, str("%.2f" % float(conf)), (int(x2), int(y2 - box_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color, 2)

        #cv2.putText(img,"Hello World!",(400,50),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),2)
        out.write(changeRGB2BGR(RGBimg))
        cv2.imshow('frame', RGBimg)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    time_end = time.time()
    time_total = time_end - time_begin
    print(NUM // time_total)

    cap.release()
    out.release()
    cv2.destroyAllWindows()