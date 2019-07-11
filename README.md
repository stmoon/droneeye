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
START Calculating mAP of sequence uav0000086_00000_v

48.99% = bicycle AP 

0.35% = motor AP 

47.88% = pedestrian AP 

23.83% = people AP 

mAP = 30.26%

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

