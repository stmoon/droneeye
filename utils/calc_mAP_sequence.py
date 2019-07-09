import time
from PIL import Image
import numpy as np
import glob
import os
import shutil
import argparse

## How to use
# modify 3 variables (data_path, result_path, sequence)
# python calc_mAP_sequence

# Follwing 3 variables should be customized.
data_path = '/media/stmoon/Data/VisDrone/Task2_Object_Detection_in_Videos/VisDrone2019-VID-val-splittest/'
result_path = '/home/ldg810/git/droneeye/m2det/output/'
# sequence = 'uav0000086_00000_v'
sequence = 'uav0000137_00458_v'
min_confidence_to_calc_mAP = 0.0

parser = argparse.ArgumentParser()
parser.add_argument('--sequence', nargs='+', type=str, help="which sequence to analyse")
args = parser.parse_args()

if args.sequence is not None:
	sequence = args.sequence[0]

print("START Calculating mAP of sequence " + sequence)

try:
	os.mkdir(os.path.dirname(os.path.realpath(__file__)) + '/mAP/input')
except FileExistsError:
	print("input folder already exist (OK)")

output_path = os.path.dirname(os.path.realpath(__file__)) + '/mAP/input/'
categories = ['','pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','']

try:
	shutil.rmtree(output_path + 'ground-truth')
except FileNotFoundError:
	print("No ground-truth folder pre-exist")
os.mkdir(output_path + 'ground-truth')
print('Generating ground-truth txt files...')
f = open(data_path + 'annotations/' + sequence + '.txt', 'r')
gtData = []
while True:
    line = f.readline()
    if not line:
    	break
    line = line.rstrip()
    line_data = line.split(',')
    
    frame_index = int(line_data[0])
    bbox_left= int(line_data[2])
    bbox_top = int(line_data[3])
    img_width = int(line_data[4])
    img_height = int(line_data[5])
    object_category = categories[int(line_data[7])]
    if int(line_data[7]) >= 1 and int(line_data[7]) <= 10:
	    output_file = open(output_path + 'ground-truth/{:07d}.txt'.format(frame_index),'ta')
	    output_line = object_category + ' ' + str(bbox_left) + ' ' + str(bbox_top) + ' ' + str(bbox_left + img_width) + ' ' + str(bbox_top + img_height) + '\n'
	    output_file.write(output_line)
	    output_file.close()
f.close()

try:
	shutil.rmtree(output_path + 'detection-results/')
except FileNotFoundError:
	print("No detection-results folder pre-exist")
os.mkdir(output_path + 'detection-results')
print('Generating detection-result txt files...')
f = open(result_path + sequence + '.txt', 'r')
ourData = []
while True:
    line = f.readline()
    if not line:
    	break
    line = line.rstrip()
    line_data = line.split(',')
    
    frame_index = int(line_data[0])
    bbox_left= int(line_data[2])
    bbox_top = int(line_data[3])
    img_width = int(line_data[4])
    img_height = int(line_data[5])
    confidence = line_data[6]
    object_category = categories[int(line_data[7])]
    if int(line_data[7]) >= 1 and int(line_data[7]) <= 10 and float(confidence) >= min_confidence_to_calc_mAP:
	    output_file = open(output_path + 'detection-results/{:07d}.txt'.format(frame_index),'ta')
	    output_line = object_category + ' ' + confidence + ' ' + str(bbox_left) + ' ' + str(bbox_top) + ' ' + str(bbox_left + img_width) + ' ' + str(bbox_top + img_height) + '\n'
	    output_file.write(output_line)
	    output_file.close()
f.close()

print('Calculating mAP...')
os.system('python '+os.path.dirname(os.path.realpath(__file__))+'/mAP/main.py -na -np')