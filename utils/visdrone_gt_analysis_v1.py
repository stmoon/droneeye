import math
import numpy
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

num_category = 12

data = []
for i in range(0,num_category):
	data.append([])

files = [f for f in listdir('/data1/VisDrone2019Data/VisDrone2019-VID-train/annotations/') if isfile(join('/data1/VisDrone2019Data/VisDrone2019-VID-train/annotations/', f))]

for file in files:
	f = open("/data1/VisDrone2019Data/VisDrone2019-VID-train/annotations/" + file, 'r')
	while True:
	# for i in range(1, 11):
	    line = f.readline()
	    if not line:
	    	break
		line = line.rstrip()
	    line_data = line.split(',')
	    
	    # frame_index = line_data[0]
	    # target_id = line_data[1]
	    # bbox_left= line_data[2]
	    # bbox_top = line_data[3]
	    img_width = int(line_data[4])
	    img_height = int(line_data[5])
	    object_category = int(line_data[7])
	    # truncation = line_data[8]
	    # occlusion = line_data[9]
	    data[object_category].append(math.sqrt(img_height * img_width))
	f.close()

# Cat 0 : ignored regions
# Cat 1 : pedestrian
# Cat 2 : people
# Cat 3 : bicycle 
# Cat 4 : car
# Cat 5 : van
# Cat 6 : truck
# Cat 7 : tricycle
# Cat 8 : awning-tricycle
# Cat 9 : bus
# Cat 10 : motor
# Cat 11 : others

categories = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']

for i in range(0,num_category):
# for i in range(0,2):
	print('# of category ' + str(i) + ' : ' + str(len(data[i])))
	bins = numpy.arange(min(data[i]), 400, 10)
	hist,bins = numpy.histogram(data[i], bins)
	hist_sum = sum(hist)
	hist = hist/float(hist_sum)
	plt.plot(bins[0:(len(hist))],hist)
	categories[i] = categories[i] + ': ' + "{:,}".format(len(data[i]))
	# print(hist)
	# print(bins)

plt.legend(categories, loc='upper right')
plt.xlabel('sqrt(Height*Width)')
plt.ylabel('num/sum(num)')
plt.show()

