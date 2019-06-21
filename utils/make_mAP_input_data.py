import time
from PIL import Image
import numpy as np
import glob

data_path = '/media/stmoon/Data/VisDrone/Task2_Object_Detection_in_Videos/VisDrone2019-VID-val/'
result_path = '/home/ldg810/git/droneeye/m2det/output/'
output_path = '/home/ldg810/git/mAP/input/'
sequence = 'uav0000137_00458_v'

categories = ['','pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','']

f = open(data_path + 'annotations/' + sequence + '.txt', 'r')
gtData = []
while True:
    line = f.readline()
    if not line:
    	break
    line = line.rstrip()
    line_data = line.split(',')
    
    frame_index = int(line_data[0])
    # target_id = line_data[1]
    bbox_left= int(line_data[2])
    bbox_top = int(line_data[3])
    img_width = int(line_data[4])
    img_height = int(line_data[5])
    object_category = categories[int(line_data[7])]
    # truncation = line_data[8]
    # occlusion = line_data[9]
    # data[object_category].append(math.sqrt(img_height * img_width))
    # gtData.append([frame_index, bbox_left, bbox_top, img_width, img_height, object_category])
    if int(line_data[7]) >= 1 and int(line_data[7]) <= 10:
	    output_file = open(output_path + 'ground-truth/{:07d}.txt'.format(frame_index),'ta')
	    # print(object_category, bbox_left, bbox_top, img_width, img_height)
	    output_line = object_category + ' ' + str(bbox_left) + ' ' + str(bbox_top) + ' ' + str(bbox_left + img_width) + ' ' + str(bbox_top + img_height) + '\n'
	    output_file.write(output_line)
	    output_file.close()
f.close()


f = open(result_path + sequence + '.txt', 'r')
ourData = []
while True:
    line = f.readline()
    if not line:
    	break
    line = line.rstrip()
    line_data = line.split(',')
    
    frame_index = int(line_data[0])
    # target_id = line_data[1]
    bbox_left= int(line_data[2])
    bbox_top = int(line_data[3])
    img_width = int(line_data[4])
    img_height = int(line_data[5])
    confidence = line_data[6]
    object_category = categories[int(line_data[7])]
    # truncation = line_data[8]
    # occlusion = line_data[9]
    # data[object_category].append(math.sqrt(img_height * img_width))
    # gtData.append([frame_index, bbox_left, bbox_top, img_width, img_height, object_category])
    if int(line_data[7]) >= 1 and int(line_data[7]) <= 10:
	    output_file = open(output_path + 'detection-results/{:07d}.txt'.format(frame_index),'ta')
	    # print(object_category, bbox_left, bbox_top, img_width, img_height)
	    output_line = object_category + ' ' + confidence + ' ' + str(bbox_left) + ' ' + str(bbox_top) + ' ' + str(bbox_left + img_width) + ' ' + str(bbox_top + img_height) + '\n'
	    output_file.write(output_line)
	    output_file.close()
f.close()