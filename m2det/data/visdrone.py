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
import numpy as np
import glob
from skimage.transform import resize

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


VOC_CLASSES = ( '__background__', # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))



class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0,5)) 
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                #cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res,bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

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
    def __init__(self, root, image_sets, preproc=None, target_transform=AnnotationTransform(),
                 dataset_name='VisDrone'):
        self.root = root
        self.image_set = image_sets
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self.annopath = root+image_sets[0]+'/annotations/' 
        self.img_path = list()
        self.img_shape = (512, 512)     # TODO : set the image size automatically
        self.anno = dict()

        # Image path
        image_sets = glob.glob(root+image_sets[0]+'/sequences/*')
        for path in image_sets :
            self.img_path += glob.glob(path+'/*.jpg')

        # Label
        # <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        anno_paths = glob.glob(self.annopath + '/*.txt')
        for path in anno_paths :
            if os.path.exists(path):
                try :
                    grp_id = path.split('/')[-1].split('.')[0]
                    labels = np.loadtxt(path, delimiter=',')
                    for i in labels :
                        i = i.tolist()
                        self.insert_anno(grp_id + str(int(i[0])), i[2:6] + [i[7]])
                except Exception as err:
                    print("OS error: {0}".format(err))
        
 

    def __getitem__(self, index):

        # Image
        img_path = self.img_path[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # Label
        grp_id = img_path.split('/')[-2]
        seq_id = img_path.split('/')[-1].split(".")[0]

        #if self.target_transform is not None:
        #    target = self.target_transform(target)
        
        if self.preproc is not None:
            key = grp_id + str(int(seq_id))
            if key in self.anno :
                img, target = self.preproc(img, np.array(self.anno[key]))
            else :
                img, target = self.preproc(img, np.array([]))
        print(target.shape)
        return img, target

        '''
        img_path = self.img_path[index]
        img = np.array(Image.open(self.img_path[index]))
        
        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #  <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        #---------
        
        grp_id = img_path.split('/')[-2]
        label_path = self._annopath + grp_id + ".txt"
        labels = None
        if os.path.exists(label_path):
            try :
                labels = np.loadtxt(label_path, delimiter=',', usecols=tuple(range(8))).reshape(-1,8)
            except:
                print("ERROR:", label_path)
        else :
            print("Not exist", label_path)

        
        # Fill matrix
        # input : left, top, width, height (in based on image, not scaled) for visdrone
        # output: scaled center x, scaled center y, scaled width, scaled height ( 0.0 ~ 1.0)
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            # Extract coordinates for unpadded + unscaled image
            x1 = labels[:,0]
            y1 = labels[:,1]
            x2 = labels[:,0] + labels[:,2]
            y2 = labels[:,1] + labels[:,3]
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            #print(labels[0:2, 0:4])
            #print(x1[:2], x2[:2], y1[:2], y2[:2])
            labels[:, 0] = ((x1 + x2) / 2) / padded_w
            labels[:, 1] = ((y1 + y2) / 2) / padded_h
            labels[:, 2] = labels[:, 2] / padded_w
            labels[:, 3] = labels[:, 3] / padded_h
            #np.set_printoptions(precision=2)
            #print(labels[0:2, 0:4])

            #print(labels[:5])
 
        num_labels = min(self.max_objects, len(labels))
        filled_labels[:num_labels,0] = labels[:num_labels,5] 
        filled_labels[:num_labels,1:5] = labels[:num_labels,0:4] 
        #filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        return input_img, filled_labels
        '''

        #########################################################################
        '''
        print(">>>> indxe :", index)
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)


        if self.preproc is not None:
            img, target = self.preproc(img, target)
            #print(img.size())

                    # target = self.target_transform(target, width, height)
        #print(target.shape)

        return img, target
        '''

    def __len__(self):
        return len(self.img_path)

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
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

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
        pass


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
