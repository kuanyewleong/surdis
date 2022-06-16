# SurDis: A Surface Discontinuity Dataset for Wearable Technology to Assist Blind Navigation in Urban Environments
## (Under Review at NeurIPS 2022 Datasets and Benchmarks Track)
The Surface Discontinuity Dataset contains over 200 sets of depth maps and their corresponding stereo image sequences (in bitmap file format to preserve their best resolution) of surface discontinuity along the walkway within Malaysia urbans, captured by our stereo vision prototype for a research study concerning navigation for the blind and low vision people.

Most of the publicly available datasets that can be applied for blind navigation have predominantly focused on solving the problem of obstacles detection (either static objects such as lamp posts and traffic signs, or dynamic objects such as vehicles and pedestrians) above the surface of a pathway. However, there is very limited research done for blind navigation in negotiating surface discontinuity at outdoor urban areas, let alone publicly available dataset that tailored for such research. Even if such datasets are available, they might not be suitable for local usages as each country has its unique built environment and standards. Additionally, most visual based technologies in recent years have focused on the mentioned issues of objects detection, or a more global problem of wayfinding. Taking into account these gaps, we present SurDis - the Surface Discontinuity Dataset of urban areas in Malaysia.

![alt text](https://github.com/kuanyewleong/surdis/fig1_1.png "sample")

# Dataset Description
These samples of surface condition of some pathways were collected from 10 different locations within Malaysia. The sampling method employed was a judgmental sampling, which is a non-probability method based on judgement that certain areas could have more samples as compared to others. Data were collected during sunny days under direct sunlight or indirect sunlight based on the locations. The recording tasks were performed under natural (uncontrolled) environment hence some data might contain anonymized passers-by or vehicles.

# Download
Cick here to download the depth maps or the bitmap stereo images of this dataset. 

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
