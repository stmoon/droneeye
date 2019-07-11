# Drone Eye

## Target

- Task 2: Object Detection in Videos in VisDrone

## Trainig Data Anaylsis
The Traning data is same with 2018 version.
![text](https://github.com/stmoon/droneeye/blob/master/docs/VisDrone2019_data_analysis.png)

## Result Format

Both the ground truth annotations and the submission of results on test data have the same format for object detection in videos. That is, each text file stores the detection results of the corresponding video clip, with each line containing an object instance in the video frame. The format of each line is as follows:

```
<frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
```

Please find the example format of the submission of results for object detection in videos here (BaiduYun|Google Drive)

[Details Link](http://aiskyeye.com/views/getInfo?loc=6)

## Sample Result (20190613)
- Green box : Ground Truth
- Red box : our result
![text](https://github.com/stmoon/droneeye/blob/master/docs/sample_result_20190613.png)

## Sample mAP Result (20190711) for a val sequence
START Calculating mAP of sequence uav0000137_00458_v (without split)

8.83% = bicycle AP 

63.49% = car AP 

22.67% = motor AP 

29.45% = pedestrian AP 

33.63% = people AP 

4.38% = van AP 

mAP = 27.07%

![text](https://github.com/stmoon/droneeye/blob/master/docs/wosplit.png)

START Calculating mAP of sequence uav0000137_00458_v (with split)

26.48% = bicycle AP 

50.55% = car AP 

28.09% = motor AP 

30.07% = pedestrian AP 

43.15% = people AP 

19.91% = van AP 

mAP = 33.04%


![text](https://github.com/stmoon/droneeye/blob/master/docs/nms_w_split.png)

  
## How to Use It

### Set up the test environment

Before the setting the environment, please use ```virtualenv``` and then install necessary libraries like below
```
pip install -r requirement
```


### Submit the result

To submit the result, please use ```test.py``` like below.

```bash
$ python test.py
```

