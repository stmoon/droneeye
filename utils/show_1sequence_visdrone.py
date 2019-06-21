import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import time
from PIL import Image
import numpy as np
import glob

data_path = '/media/stmoon/Data/VisDrone/Task2_Object_Detection_in_Videos/VisDrone2019-VID-val/'
result_path = '/home/ldg810/git/droneeye/m2det/output/'
# sequence = 'uav0000086_00000_v'
sequence = 'uav0000137_00458_v'
confidence_threshold = 0.1

max_frame = 1
for filename in glob.iglob(data_path + 'sequences/' + sequence + '/*.jpg',recursive=True):
	cur_frame = int(filename.split('/')[-1][0:-4])
	if max_frame < cur_frame:
		max_frame = cur_frame

plt.rcParams["figure.figsize"] = (20,15)
fig,ax = plt.subplots(1)

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
    object_category = int(line_data[7])
    # truncation = line_data[8]
    # occlusion = line_data[9]
    # data[object_category].append(math.sqrt(img_height * img_width))
    gtData.append([frame_index, bbox_left, bbox_top, img_width, img_height, object_category])
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
    confidence = float(line_data[6])
    object_category = int(line_data[7])
    # truncation = line_data[8]
    # occlusion = line_data[9]
    # data[object_category].append(math.sqrt(img_height * img_width))
    ourData.append([frame_index, bbox_left, bbox_top, img_width, img_height, object_category, confidence])
f.close()
	
def animate(i):
	frame = i+1

	if frame > max_frame:
		exit(0)
	im = np.array(Image.open(data_path + 'sequences/' + sequence + '/{:07d}.jpg'.format(frame)), dtype=np.uint8)
	ax.clear()
	ax.imshow(im)

	for data in gtData:
		if data[0] == frame:
			rect = patches.Rectangle((data[1],data[2]),data[3],data[4],linewidth=2,edgecolor='chartreuse',facecolor='none')
			ax.text(data[1] + 3, data[2] - 5, str(data[5]), color='chartreuse', fontsize=13, fontweight='bold')
			ax.add_patch(rect)

	for data in ourData:
		if data[0] == frame and data[6] >= confidence_threshold:
			rect = patches.Rectangle((data[1],data[2]),data[3],data[4],linewidth=1,edgecolor='r',facecolor='none')
			ax.text(data[1] + 3, data[2] - 5, str(data[5]), color='r', fontsize=13, fontweight='bold')
			ax.add_patch(rect)


ani = animation.FuncAnimation(fig, animate, interval=500)

plt.show()