from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/visdrone.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/visdrone.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/visdrone.names", help="path to class label file")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression") 
parser.add_argument("--n_cpu", type=int, default=6, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
test_path = data_config["valid"]
num_classes = int(data_config["classes"])


# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)

if cuda:
    model = model.cuda()

model.eval()

# Get dataloader
dataset = VisDroneDataset(test_path)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

all_detections = []
all_annotations = []

for batch_i, (imgs_path, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

    imgs = Variable(imgs.type(Tensor))

    with torch.no_grad():
        outputs = model(imgs)
        outputs = non_max_suppression(outputs, num_classes, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

    for file_path, output, annotations in zip(imgs_path,outputs, targets):

        all_detections.append([np.array([]) for _ in range(num_classes)])
        if output is not None:
            # Get predicted boxes, confidence scores and labels
            pred_boxes = output[:, :5].cpu().numpy()
            scores = output[:, 4].cpu().numpy()
            pred_labels = output[:, -1].cpu().numpy()

            # Order by confidence
            sort_i = np.argsort(scores)
            pred_labels = pred_labels[sort_i]
            pred_boxes = pred_boxes[sort_i]

            for label in range(num_classes):
                all_detections[-1][label] = pred_boxes[pred_labels == label]

            ## output
            _, fp =  os.path.split(file_path)
            frame_index = os.path.splitext(fp)[0].split('_')[-1]
            
            print(file_path)
            print(file_path.split('/')[-2])
            output_file = open('output/'+file_path.split('/')[-2]+'.txt','a')
            for i in range(len(pred_boxes)) :
                result = []
                result.append(int(frame_index))
                result.append(-1)
                result += list(map(int,pred_boxes[i].tolist()[:-1]))
                result.append(round(scores[i],4))
                result.append(int(pred_labels[i]))
                result.append(-1)
                result.append(-1)
                output_file.write(','.join(str(x) for x in result) + '\n')
#                print(result)
            output_file.close()
