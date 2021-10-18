# Sptiotemporal model for action localization


## Process
1. the architecture is a single-stage network with two branches.
2. One branch extracts the spatial features of the key frame via a 2D-CNN
3. the other branch models the spatiotemporal features of the clip consisting of previous frames via a 3D-CNN
4. a channel fusion and attention mechanism is used to aggregate these features smoothly
5. Finally, frame-level detections are produced using the fused features, and provide a linking algorithm to generate action tubes.

## 1. Clone the project

```bash
git clone git@github.com:wiki-yu/spatio-temporal-action-localization.git
```

## 2. Enviroment Setting

### Create virtual enviroment

Create Virtual (Windows) Environment:
```shell script
py -m venv env
.\env\Scripts\activate
```

Create Virtual (Linux/Mac) Environment:
```shell script
py -m venv env
source env/bin/activate
```

### Install packages
```shell script
pip3 install -r requirements.txt
```
```shell script
pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## 3. Data Preparation
**training & testing datasets:**
1. There are three classes currently: drawer, clean_tray, count_pills
2. Please download the image data under the file folder /dataset/ucsp/
3. Data link: [\[Teams drive\]](https://teams.microsoft.com/_#/files/IAI-AI?threadId=19%3A2887ad0aaac040a1b7ad4681f0b867be%40thread.tacv2&ctx=channel&context=rgb-images&rootfolder=%252Fsites%252FFiiUSA-iAIGroup-IAI-AI%252FShared%2520Documents%252FIAI-AI%252FAction%2520Localization%252FUCSP%2520annotation%252Fspatio_temporal_annotation%252Frgb-images)

**model files**
1. Download the weights file folder at [\[Teams drive\]](https://teams.microsoft.com/_#/files/IAI-AI?threadId=19%3A2887ad0aaac040a1b7ad4681f0b867be%40thread.tacv2&ctx=channel&context=weights&rootfolder=%252Fsites%252FFiiUSA-iAIGroup-IAI-AI%252FShared%2520Documents%252FIAI-AI%252FAction%2520Localization%252FSpatio-temporal%2520action%2520localization%252Fweights) and put it under the project root folder. 
2. Download the trained model and put it under the project root folder:
Model link: [\[Teams drive\]](https://teams.microsoft.com/_#/files/IAI-AI?threadId=19%3A2887ad0aaac040a1b7ad4681f0b867be%40thread.tacv2&ctx=channel&context=backup&rootfolder=%252Fsites%252FFiiUSA-iAIGroup-IAI-AI%252FShared%2520Documents%252FIAI-AI%252FAction%2520Localization%252FSpatio-temporal%2520action%2520localization%252Fbackup)

## 4. Training
python3 mytrain.py

## 5. Inference

```shell script
python3 test_images.py
```
```shell script
python3 test_video.py
```








