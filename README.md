# SurDis: A Surface Discontinuity Dataset for Wearable Technology to Assist Blind Navigation in Urban Environments
### (Under Review at NeurIPS 2022 Datasets and Benchmarks Track)
The Surface Discontinuity Dataset contains over 200 sets of depth maps and their corresponding stereo image sequences (in bitmap file format to preserve their best resolution) of surface discontinuity along the walkway within Malaysia urbans, captured by our stereo vision prototype for a research study concerning navigation for the blind and low vision people.

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
Click here to download the dataset: [SurDis](https://1drv.ms/u/s!AkMf6DxiFnMnvQCWjMMow4hks5Py?e=VAzQCe).

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

# License
This work (inclusive the contents on this site and the dataset) is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License and is intended for non-commercial uses. If you are interested in using the dataset for commercial purposes please contact us.

# Example Usages
We propose the following usages which are relevant to developing a wearable assistive technology for the blind and low vision people:
- developing of wearable assistive tool that detects surface discontinuity in near real-time
- including of SurDis into other datasets that have various urban objects to train a more diverse object detection model targeting urban navigation
- utilizing of the depth map to extract distance information that can be supplied via the feedback mechanism of an assistive tool for blind navigation
- designing of evaluation system that rates the level of hazard for each class or each instance of surface discontinuity, this can become a hazard alert mechanism for an assistive tool in blind navigation

# Recommended Practice for Object Detection Model 
As a measure to control the quality of our annotation, we annotated the surface discontinuities with tight bounding boxes. Thus, it is reccommended that when you are training your model using this dataset, you may slightly expand the bounding boxes (i.e. add a few units of pixel to all edges). This extra context might help your models to better recognize the objects when background pixels around the borders are included. 
