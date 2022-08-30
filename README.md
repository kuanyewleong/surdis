# SurDis: A Surface Discontinuity Dataset for Wearable Technology to Assist Blind Navigation in Urban Environments
### (Under Review at NeurIPS 2022 Datasets and Benchmarks Track)
The Surface Discontinuity Dataset contains over 200 sets of depth map sequences and their corresponding stereo image sequences (in bitmap file format to preserve their best resolution) of surface discontinuity along the walkway within Malaysia urbans, captured by our stereo vision prototype for a research study concerning navigation for the blind and low vision people.

Most of the publicly available datasets that can be applied for blind navigation have predominantly focused on solving the problem of obstacles detection (either static objects such as lamp posts and traffic signs, or dynamic objects such as vehicles and pedestrians) above the surface of a pathway. However, there is very limited research done for blind navigation in negotiating surface discontinuity at outdoor urban areas, let alone publicly available dataset that tailored for such research. Even if such datasets are available, they might not be suitable for local usages as each country has its unique built environment and standards. Additionally, most visual based technologies in recent years have focused on the mentioned issues of objects detection, or a more global problem of wayfinding. Taking into account these gaps, we present SurDis - the Surface Discontinuity Dataset of urban areas in Malaysia.

![alt text](https://github.com/kuanyewleong/surdis/blob/main/fig1_1.png "sample")

# Dataset Description
These samples of surface condition of some pathways were collected from 10 different locations within Malaysia. SurDis has 200 sets of depth map sequences with annotation of various surface discontinuities from 10 selected locations, captured in video recording mode by a person mimicking the walking style of a typical BLV individual. Each sequence set contains about 100 to 150 depth maps, and we generated a total of 17302 such depth maps. We also provide the original stereo image sequences corresponding to the depth maps. The generated depth maps are stored as numpy arrays (in .npy format), and they are generated based on a disparity mapping technique as given in the [script](https://github.com/kuanyewleong/surdis/blob/main/util/disparity_mapping.py). The stereo images are stored as monochromatic bitmap format.

The sampling method employed was a judgmental sampling, which is a non-probability method based on judgement that certain areas could have more samples as compared to others. Data were collected during sunny days under direct sunlight or indirect sunlight based on the locations. The recording tasks were performed under natural (uncontrolled) environment hence some data might contain anonymized passers-by or vehicles.

In the following figure are some examples of surface discontinuity. Hazardous conditions are indicated by the red arrows, counterclockwise: partially covered drainage next to some steps that connect the walkway, uncovered drainage between a steps connecting the road and the aisle, high altitude drop-off leading to uncovered drainage at the edge of walkway, uncovered drainage between a ramp and some steps, drop-off along the edge of a walkway without railing, and blended gradients between a ramp and steps.
![alt text](https://github.com/kuanyewleong/surdis/blob/main/fig1_2.png "sample")

# Data Class Label
Based on the physical attributes of the collected data, there are 5 distinctive classes of surface discontinuity: (1) down-steps, (2) up-steps, (3) uncovered drainage, (4) drop-off without handrail, and (5) mixed gradient. 

# Dataset Download Link and File Structure
Click here to download the dataset: [SurDis](https://1drv.ms/u/s!AkMf6DxiFnMnvQCU1ve8cJVSqL4G?e=UcFEv4). 
(Please note that we are currently double-checking the anonymizing task for some of the raw images before the final release of the dataset to Zenodo site with a DOI. Thus, the current location is a temporary solution.)

After decompressing the downloaded folder from the above link, the structure of the dataset will be:
```
.
└── $Data_root/
    ├── annotation_files/
    │   ├── PASCAL_VOC
    │   └── COCO_JSON
    ├── depth_maps/
    │   ├── trainset
    │   └── testset
    └── bitmap_stereo_images/
        ├── trainset/
        │   ├── left-image-folder
        │   └── right-image-folder
        └── testset/
            ├── left-image-folder
            └── right-image-folder
 
 ```
For simplicity of usage, we made all the annotation of bounding boxes based on the file path of left images. If you are training with depth maps, you can change the file path to point to the .npy files in the "$Data_root/depth_maps/trainset" directory.

### Files and Naming Convention
Depth maps in "depth_maps/trainset/depth_x" are generated from image pairs in "bitmap_stereo_images/trainset/left-image-folder/left_x" and "bitmap_stereo_images/trainset/right-image-folder/right_x".

For example, a depth map file in *depth_maps/trainset/depth_1/map_113.npy* is generated from stereo image pair in *bitmap_stereo_images/trainset/left-image-folder/left_1/l_img_113.bmp* and *bitmap_stereo_images/trainset/right-image-folder/right_1/r_img_113.bmp*.

Please refer the example data in repository [example_data](https://github.com/kuanyewleong/surdis/tree/main/example_data) for better understanding. 


# Potential Usages
We propose the following usages which are relevant to developing a wearable assistive technology for the blind and low vision people:
- developing of wearable assistive tool that detects surface discontinuity in near real-time
- including of SurDis into other datasets that have various urban objects to train a more diverse object detection model targeting urban navigation
- utilizing of the depth map to extract distance information that can be supplied via the feedback mechanism of an assistive tool for blind navigation
- designing of evaluation system that rates the level of hazard for each class or each instance of surface discontinuity, this can become a hazard alert mechanism for an assistive tool in blind navigation

# Tutorial: Loading Data and Training Models

```python
from __future__ import division, print_function, unicode_literals
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import matplotlib.pyplot as plt
%matplotlib inline
plt.ion()
import xml.etree.ElementTree as ET
from models import *
import torch.nn as nn
import torch.optim as optim
from glob import glob
import os,sys

# (this is based on a simple Resnet for simplicity, 
# you may experiment with other models from the models file)
resnet_input = 225 # change this for other model
```


```python
# set some hyper-parameters
batch_size = 128
num_epochs = 100
learning_rate =  0.005
hyp_momentum = 0.9
```

## Getting the data ready
Use the following links to locally download the data:
<br/>https://1drv.ms/u/s!AkMf6DxiFnMnvQCU1ve8cJVSqL4G?e=UcFEv4
<br/> Prepare the data as follows (example for Pascal VOC):
<br/> For every sample in the dataset, extract/crop the object patch from the depth map one by one using their respective co-ordinates:[xmin, ymin, xmax, ymax], resize the map to resnet_input, and store it with its class label information. Do the same for training/validation and test datasets. <br/>


```python
classes = ('__background__',
           'down_steps', 'up_steps', 'uncovered_drainage',
           'drop_off','mixed_gradient'
           )
```

```python
class voc_dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, train, transform=None):
        self.train = train
        self.transforms = transform
        self.data = []
        self.objects = []
        count_pclass = [0 for i in range(5)]
        
        if(train==True):
            with open(root_dir + "annotation_files/PASCAL_VOC/trainset/trainval.txt") as f:
                for l in f:
                    self.data.append(l.split())
        else:
            with open(root_dir + "annotation_files/PASCAL_VOC/testset/test.txt") as f:
                for l in f:
                    self.data.append(l.split())
        
        for f in self.data:
            tree = ET.parse( root_dir + "annotation_files/PASCAL_VOC" + f[0] + ".xml")
            filename = tree.find('filename').text
            for obj in tree.findall('object'):
                obj_struct = {}
                if(obj.find('name').text in classes):
                    obj_struct['img_name'] = filename
                    obj_struct['class'] = classes.index(obj.find('name').text)
                    bbox = obj.find('bndbox')
                    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                          int(bbox.find('ymin').text),
                                          int(bbox.find('xmax').text),
                                          int(bbox.find('ymax').text)]
                    self.objects.append(obj_struct)
                    
        
    def __len__(self):
        return len(self.objects)
        
    def __getitem__(self, idx):
        f_name = self.objects[idx]['depth_name']
        clss = self.objects[idx]['class']
        if (self.train == True) :
            depth = torch.from_numpy(np.load(('depth_maps/trainset/' + f_name))
        else:
            depth = torch.from_numpy(np.load(('depth_maps/testset/' + f_name))
        boundary_box = self. objects[idx]['bbox']
        area = (boundary_box[0], boundary_box[1], boundary_box[2], boundary_box[3])
        cr_depth = depth
        if self.transforms is not None:
            cr_depth = depth.crop(area)
            cr_depth = self.transforms(cr_image)
        return cr_depth, clss
            
```

## Train the netwok
<br/>You can now train the network on the prepared dataset from above.
<br/>First, we do some normalization and then load the data into loader with Torch Dataloader


```python
# composed_transform = transforms.Compose([transforms.Scale((resnet_input,resnet_input)),
#                                          transforms.ToTensor(),
#                                          transforms.RandomHorizontalFlip()])
transformations = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.465], [0.291])]) # based on our own calculation
train_dataset = voc_dataset(root_dir='depth_maps/trainset/', train=True, transform=transformations) # Supply proper root_dir
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
```

### Define model and other training parameters:

```python
device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
num_class = 5
model = models.resnet10(num_class)
model.to(device)
cudnn.benchmark = True  
```


```python
criterion = nn.CrossEntropyLoss()
# Update if any errors occur
optimizer = optim.SGD(model.parameters(), learning_rate, hyp_momentum)

```


```python
def train(curr_epoch):
    model.train()
    running_loss = 0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        
        top_p, top_class = logps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        
        loss = criterion(logps, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    
        running_corrects += torch.mean(equals.type(torch.FloatTensor)).item()

        
    train_epoch_loss = running_loss/len(train_loader)
    train_acc = running_corrects /len(train_loader)
    
    test_loss = 0
    test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = nputs.to(device),labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    test_epoch_loss = test_loss/len(test_loader)
    test_acc = test_accuracy/len(test_loader)

    print('{} Train_Loss: {:.4f} Train_Acc: {:.4f} Test_Loss: {:.4f} Test_Acc: {:.4f}'.format(
         curr_epoch,train_epoch_loss, train_acc, test_epoch_loss, test_acc))
    print("-------------------------")
            
    torch.save(model, 'one_layer_model.pth')
```


```python
for curr_epoch in range(num_epochs):
    train(curr_epoch)
```


# Testing and Accuracy Calculation
For detection, we adopt a slding window method to test the above trained model:<br/>
Take some windows of varying size and aspect ratios and slide it through the test image (considering some stride of pixels) from left to right, and top to bottom, detect the class scores for each of the window, and keep only those which are above a certain threshold value. 


```python
def sliding_window(width,height):
    box_dim =[[128,128],[200,200],[400,400],[180,360],[90,180],[180,90]]
    fe_size = (800//40)
    ctr_x = np.arange(16, (fe_size+1) * 16, 16)
    ctr_y = np.arange(16, (fe_size+1) * 16, 16)
    ctr = np.zeros((len(ctr_x)*len(ctr_y),2))

    index = 0
    for x in range(len(ctr_x)):
        for y in range(len(ctr_y)):
            ctr[index, 1] = ctr_x[x] - 8
            ctr[index, 0] = ctr_y[y] - 8
            index +=1
    boxes = np.zeros(((fe_size * fe_size * 9), 4))
    index = 0
    for c in ctr:
        ctr_y, ctr_x = c
        for i in range(len(box_dim)):
            h = box_dim[i][0]
            w = box_dim[i][1]
            boxes[index, 0] = ctr_x - w / 2.
            boxes[index, 1] = ctr_y - h / 2.
            boxes[index, 2] = ctr_x + w / 2.
            boxes[index, 3] = ctr_y + h / 2.
            index += 1
    bbox = np.asarray([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=np.float32) # [y1, x1, y2, x2] format
    labels = np.asarray([1, 5], dtype=np.int8) # 0 represents backgrounda
    index_inside = np.where(
            (boxes[:, 0] >= 0) &
            (boxes[:, 1] >= 0) &
            (boxes[:, 2] <= width) &
            (boxes[:, 3] <= height)
        )[0]
    label = np.empty((len(index_inside), ), dtype=np.int32)
    label.fill(-1)
    valid_boxes = boxes[index_inside]
    return valid_boxes
```

Apply non_maximum_supression to reduce the number of boxes. You may experiment with the threshold value for non maximum supression between [0,1].


```python
def non_maximum_supression(boxes,threshold = 0.5):
    if len(boxes) == 0:
        return []

    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in range(0, last):
            j = idxs[pos]

            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[j]

            if overlap > threshold:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return boxes[pick]
```


```python

# trans1 = transforms.ToPILImage()
trans = transforms.ToTensor()

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

```


```python
model = torch.load('save_model.pth')
criterion = nn.CrossEntropyLoss()
# Update if any errors occur
optimizer = optim.SGD(model.parameters(), learning_rate, hyp_momentum)
```

## Get the Ground Truths Bounding Boxes for Evaluation


```python
data = []
with open("depth_maps/testset/" + "depth_maps/testset/test.txt") as f:    
    for l in f:
        data.append(l.split())

count_pclass = [0 for i in range(5)]

ground_truth_boxes = [] 
for f in data:
    G = []
    tree = ET.parse( "depth_maps/testset/" + "annotation_files/PASCAL_VOC/" + f[0] + ".xml")
    filename = tree.find('filename').text
    for obj in tree.findall('object'):
        if(obj.find('name').text in classes):
            bbox = obj.find('bndbox')
            obj_struct = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            G.append(obj_struct)



        if(obj.find('name').text in back_classes):
            if(count_pclass[back_classes.index(obj.find('name').text)] < 100):
                count_pclass[back_classes.index(obj.find('name').text)]  +=1 
                bbox = obj.find('bndbox')
                obj_struct = [int(bbox.find('xmin').text),
                                      int(bbox.find('ymin').text),
                                      int(bbox.find('xmax').text),
                                      int(bbox.find('ymax').text)]
                G.append(obj_struct)
    ground_truth_boxes.append(G)

# ground_truths = {}
# for i,f in enumerate(data):
#     ground_truths[f[0]] = ground_truth_boxes[i] 
```

# Test Dataset


```python
Test_dataset = []
for f in data:
    depth = torch.from_numpy(np.load(('depth_maps/testset/' + f[0]+'.npy'))    
    Test_dataset.append(depth)

a_out = Test_dataset

Test_dataset = []
Test_dataset.append(a_out[0])
Test_dataset.append(a_out[1])
Test_dataset.append(a_out[2])
Test_dataset.append(a_out[3])
```

# Test the trained model on the test dataset.

```python
def test(model):
    results = []
    for data in Test_dataset:
        image = data
        w, h = image.size[0], image.size[1]
        boxes = sliding_window(w, h)
        res = []
        for box in boxes:
            area = (box[0], box[1], box[2], box[3])
            im = image.crop(area)
            im = transformations(im)
            im = im.unsqueeze_(0)
            im = im.to(device)
            k  = model.forward(im)
            prob = torch.nn.functional.softmax(k, dim=1)[0]
            cls = prob.data.cpu().numpy().argmax()
            if(cls!=0):
                res.append(box)

        bboxes = non_maximum_supression(np.array(res),0.5)
        results.append(bboxes)
    return results

results = test(model)

final_res = []
for res in results:
    temp = []
    for r in res:
        temp.append(list(r))
    
    final_res.append(temp)
```

# Evaluation Metrics (AP and mAP)


```python
from __future__ import absolute_import, division, print_function

from copy import deepcopy
import json
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('white')
sns.set_context('poster')
```


```python
COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou
```


```python
def get_single_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
```


```python
def calc_precision_recall(depth_results):
    """Calculates precision and recall from the set of depth maps
    Args:
        depth_results (dict): dictionary formatted like:
            {
                'depth_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'depth_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in depth_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)
```


```python
def get_model_scores_map(pred_boxes):
    """Creates a dictionary of from model_scores to depth ids.
    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'
    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)
    """
    model_scores_map = {}
    for depth_id, val in pred_boxes.items():
        for score in val['scores']:
            if score not in model_scores_map.keys():
                model_scores_map[score] = [depth_id]
            else:
                model_scores_map[score].append(depth_id)
    return model_scores_map
```


```python
def get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=0.5):
    """Calculates average precision at given IoU threshold.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (list of list of floats): list of locations of predicted
            objects as [xmin, ymin, xmax, ymax]
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: avg precision as well as summary info about the PR curve
        Keys:
            'avg_prec' (float): average precision for this IoU threshold
            'precisions' (list of floats): precision value for the given
                model_threshold
            'recall' (list of floats): recall value for given
                model_threshold
            'models_thrs' (list of floats): model threshold value that
                precision and recall were computed for.
    """
    model_scores_map = get_model_scores_map(pred_boxes)
    sorted_model_scores = sorted(model_scores_map.keys())

    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for depth_id in pred_boxes.keys():
        arg_sort = np.argsort(pred_boxes[depth_id]['scores'])
        pred_boxes[depth_id]['scores'] = np.array(pred_boxes[depth_id]['scores'])[arg_sort].tolist()
        pred_boxes[depth_id]['boxes'] = np.array(pred_boxes[depth_id]['boxes'])[arg_sort].tolist()

    pred_boxes_pruned = deepcopy(pred_boxes)

    precisions = []
    recalls = []
    model_thrs = []
    depth_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        # In 1st iter define depth_results for the first time:
        depth_ids = gt_boxes.keys() if ithr == 0 else model_scores_map[model_score_thr]
        for depth_id in depth_ids:
            gt_boxes_depth = gt_boxes[depth_id]
            box_scores = pred_boxes_pruned[depth_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    pred_boxes_pruned[depth_id]
                    start_idx += 1
                else:
                    break

            # Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[depth_id]['scores'] = pred_boxes_pruned[depth_id]['scores'][start_idx:]
            pred_boxes_pruned[depth_id]['boxes'] = pred_boxes_pruned[depth_id]['boxes'][start_idx:]

            # Recalculate results for this depth
            depth_results[depth_id] = get_single_depth_results(
                gt_boxes_depth, pred_boxes_pruned[depth_id]['boxes'], iou_thr)

        prec, rec = calc_precision_recall(depth_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs}
```


```python
def plot_pr_curve(
    precisions, recalls, category='Person', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    ax.scatter(recalls, precisions, label=label, s=5, color=color)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall curve for {}'.format(category))
    ax.set_xlim([0.0,1.3])
    ax.set_ylim([0.0,1.2])
    return ax

```


```python
iou_thr = 0.5
start_time = time.time()
data = get_avg_precision_at_iou(ground_truth_boxes, final_res, iou_thr=iou_thr)
end_time = time.time()
print('Single IoU calculation took {:.4f} secs'.format(end_time - start_time))
print('avg precision: {:.4f}'.format(data['avg_prec']))

start_time = time.time()
ax = None
avg_precs = []
iou_thrs = []
for idx, iou_thr in enumerate(np.linspace(0.5, 0.95, 10)):
    data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
    avg_precs.append(data['avg_prec'])
    iou_thrs.append(iou_thr)

    precisions = data['precisions']
    recalls = data['recalls']
    ax = plot_pr_curve(
        precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx*2], ax=ax)

# prettify for printing:
avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
print('map: {:.2f}'.format(100*np.mean(avg_precs)))
print('avg precs: ', avg_precs)
print('iou_thrs:  ', iou_thrs)
plt.legend(loc='upper right', title='IOU Thr', frameon=True)
for xval in np.linspace(0.0, 1.0, 11):
    plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
end_time = time.time()
print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
plt.show()
```

# License
This work (inclusive the contents on this site and the dataset) is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License and is intended for non-commercial uses. If you are interested in using the dataset for commercial purposes please contact us. Refer [this document](https://github.com/kuanyewleong/surdis/blob/main/License.md) for the license.

# The Team and Maintenance Effort
This project started as a govenrment funded project and now it has grown to a community-driven project with several skillful engineers and researchers contributing to it. We will put in our best effort to maintain the site and its updates.
