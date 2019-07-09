"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import imutils
import numpy as np
import glob
from skimage.transform import resize

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


# (i.e., ignored regions (0), pedestrian (1), people (2), bicycle (3), car (4), van (5), truck (6), tricycle (7), awning-tricycle (8), bus (9), motor (10), others (11))
VIC_CLASSES = ( 'ignoredregions', # always index 0
    'pedestrian', 'people', 'bicycle', 'car',
    'van', 'truck', 'tricycle', 'awning-tricycle',
    'bus', 'motor', 'others')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class VisDroneDetection(data.Dataset):

    """
    VisDrone Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    def __init__(self, root, image_sets, preproc=None,
                 dataset_name='VisDrone', split=False):
        self.root = root
        self.image_set = image_sets
        self.preproc = preproc
        #self.target_transform = target_transform
        self.name = dataset_name
        self.annopath = root+image_sets[0]+'/annotations/' 
        self.img_path = list()
        self.img_shape = (512, 512)     # TODO : set the image size automatically
        self.anno = dict()

        # Label
        # <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        anno_paths = glob.glob(self.annopath + '/*.txt')
        for path in anno_paths :
            if os.path.exists(path):
                try :
                    grp_id = path.split('/')[-1].split('.')[0]
                    labels = np.loadtxt(path, delimiter=',')
                    for i in labels :
                        object_category = i[7]
                        # change (left, top, width, height) -> (left, top, right, bottom)
                        i[4] += i[2] 
                        i[5] += i[3]
                        i = i.tolist()
                        self.insert_anno(grp_id + str(int(i[0])), i[2:6] + [i[7]])
                except Exception as err:
                    print("OS error: {0}".format(err))
        
        # Image path
        image_sets = glob.glob(root+image_sets[0]+'/sequences/*')
        for grp_path in image_sets :
            if('split' not in grp_path.split('/')[-1]):
                if split:
                    try:
                        os.mkdir(grp_path + '_split')
                    except FileExistsError:
                        print("split folder already exist (OK)")
                
                sub_path = glob.glob(grp_path+'/*.jpg')
                for path in sub_path :
                    if split:
                        # extract group and sub id
                        grp_id = path.split('/')[-2]
                        sub_id = path.split('/')[-1].split(".")[0]
                    
                        raw_image = cv2.imread(path)
                        width = raw_image.shape[0]
                        height = raw_image.shape[1]
                        cropped_lu = raw_image[0:int(width/2), 0:int(height/2)]
                        cropped_ru = raw_image[int(width/2):width, 0:int(height/2)]
                        cropped_ld = raw_image[0:int(width/2), int(height/2):height]
                        cropped_rd = raw_image[int(width/2):width, int(height/2):height]
                        lu_path = grp_path + '_split/' + sub_id + '_1.jpg'
                        ru_path = grp_path + '_split/' + sub_id + '_2.jpg'
                        ld_path = grp_path + '_split/' + sub_id + '_3.jpg'
                        rd_path = grp_path + '_split/' + sub_id + '_4.jpg'
                        cv2.imwrite(lu_path, cropped_lu)
                        cv2.imwrite(ru_path, cropped_ru)
                        cv2.imwrite(ld_path, cropped_ld)
                        cv2.imwrite(rd_path, cropped_rd)

                        self.img_path += [path, lu_path, ru_path, ld_path, rd_path]
                    else:
                        self.img_path += [path]

                #     # check annotation and add image if the image has annotation
                #     key = grp_id + str(int(sub_id))
                #     if key in self.anno :
                #         self.img_path += [path]
                #     else :
                #         print('not exist', key)

    def __getitem__(self, index):

        # Image
        img_path = self.img_path[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # Label
        grp_id = img_path.split('/')[-2]
        seq_id = img_path.split('/')[-1].split(".")[0]

        if self.preproc is not None:
            key = grp_id + str(int(seq_id))
            if key in self.anno :
                # remove the area using unused labels (0: ignoredregions, 11: others)
                for i in self.anno[key].copy() : 
                    i = list(map(int, i))

                    if i[4] == 0 or i[4] == 11 :
                        u = tuple([img[:,:,ix].mean() for ix in range(3)])
                        u = tuple(map(int,u))
                        cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),u,-1)
                        self.anno[key].remove(i)

                img, target = self.preproc(img, np.array(self.anno[key]))
            else :
                print("EMPTY: ", key, flush=True)
                img, target = self.preproc(img, np.array([]))
        else :
            print("ERROR:  There is no preproc method")

        return img, target

    def __len__(self):
        return len(self.img_path)

    def summary_anno(self, target, title="TEST") :
        total = {}
        for t in target :
            cls = int(t[-1])
            if cls in total : 
                total[cls] +=  1
            else :
                total[cls] =  1
        print(title, ': ', sorted(total.items()))

    def insert_anno(self, key, value) :
        if not key in self.anno :
            self.anno[key] = [value]
        else :
            self.anno[key].append(value)


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_path = self.img_path[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        return img
        # img_id = self.ids[index]
        # return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt
        '''
        pass


    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        '''
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        '''
        self._write_vis_results_file(all_boxes)

        pass

    def _write_vis_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(VIC_CLASSES):
            cls_ind = cls_ind 
            if cls == 'ignoredregions' or cls == 'others':
                continue
            print('Writing {} VIS results file'.format(cls))
            for im_ind, img_path in enumerate(self.img_path):
                sequence_name = img_path.split('/')[-2]
                if 'split' in sequence_name:
                    # 1 : lu, 2 : ld, 3: ru, 4: rd
                    image = cv2.imread(img_path)
                    width = image.shape[1]
                    height = image.shape[0]
                    split_offset_width = [0, 0, 0, width, width]
                    split_offset_height = [0, 0, height, 0, height]

                    # print('split here!', img_path)
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        frame_index = int(img_path.split('/')[-1][0:-6])
                        split_index = int(img_path.split('/')[-1][-5:-4])
                        score = dets[k, -1]
                        object_category = cls_ind
                        bbox_left = int(dets[k, 0] + 1)
                        bbox_top = int(dets[k, 1] + 1)
                        bbox_width = int((dets[k, 2] + 1) - bbox_left)
                        bbox_height = int((dets[k, 3] + 1) - bbox_top)
                        # print('split data : ', frame_index, split_index, split_offset_width[split_index], split_offset_height[split_index])
                        f.write('{:d},-1,{:d},{:d},{:d},{:d},{:.1f},{:d},-1,-1\n'
                            .format(frame_index, bbox_left + split_offset_width[split_index], bbox_top + split_offset_height[split_index], bbox_width, bbox_height, score, object_category))
                        # dets[k, -1] : confidence
                        # dets[k, 0] + 1 : left
                        # dets[k, 1] + 1 : top
                        # dets[k, 2] + 1 : right
                        # dets[k, 3] + 1 : bottom
                else:
                    f = open('output/'+img_path.split('/')[-2]+'.txt','ta')

                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        frame_index = int(img_path.split('/')[-1][0:-4])
                        score = dets[k, -1]
                        object_category = cls_ind
                        bbox_left = int(dets[k, 0] + 1)
                        bbox_top = int(dets[k, 1] + 1)
                        bbox_width = int((dets[k, 2] + 1) - bbox_left)
                        bbox_height = int((dets[k, 3] + 1) - bbox_top)
                        f.write('{:d},-1,{:d},{:d},{:d},{:d},{:.1f},{:d},-1,-1\n'
                            .format(frame_index, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category))
                        # dets[k, -1] : confidence
                        # dets[k, 0] + 1 : left
                        # dets[k, 1] + 1 : top
                        # dets[k, 2] + 1 : right
                        # dets[k, 3] + 1 : bottom

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
