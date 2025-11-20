# Images as Noisy Labels: Unleashing the Potential of the Diffusion Model for Open-Vocabulary Semantic Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Static Badge](https://img.shields.io/badge/View-Poster-purple)](https://iccv.thecvf.com/virtual/2025/poster/645)
[![Static Badge](https://img.shields.io/badge/Pub-ICCV'25-red)](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Images_as_Noisy_Labels_Unleashing_the_Potential_of_the_Diffusion_ICCV_2025_paper.html)
[![Static Badge](https://img.shields.io/badge/View-Project-green)](https://fanlihub.github.io/DEDOS/)

This repository is the official PyTorch implementation of the **ICCV 2025** (**Highlight**) paper:
[Images as Noisy Labels: Unleashing the Potential of the Diffusion Model for Open-Vocabulary Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Images_as_Noisy_Labels_Unleashing_the_Potential_of_the_Diffusion_ICCV_2025_paper.html),
authored by Fan Li, Xuanbin Wang, Xuan Wang, Zhaoxiang Zhang, and Yuelei Xu.

**Abstract:**
Recently, open-vocabulary semantic segmentation has garnered growing attention. Most current methods leverage vision-language models like CLIP to recognize unseen categories through their zero-shot capabilities. However, CLIP struggles to establish potential spatial dependencies among scene objects due to its holistic pre-training objective, causing sub-optimal results. In this paper, we propose a DEnoising learning framework based on the Diffusion model for Open-vocabulary semantic Segmentation, called DEDOS, which is aimed at constructing the scene skeleton. Motivation stems from the fact that diffusion models incorporate not only the visual appearance of objects but also embed rich scene spatial priors. Our core idea is to view images as labels embedded with "noise"--non-essential details for perceptual tasks--and to disentangle the intrinsic scene prior from the diffusion feature during the denoising process of the images. Specifically, to fully harness the scene prior knowledge of the diffusion model, we introduce learnable proxy queries during the denoising process. Meanwhile, we leverage the robustness of CLIP features to texture shifts as supervision, guiding proxy queries to focus on constructing the scene skeleton and avoiding interference from texture information in the diffusion feature space. Finally, we enhance spatial understanding within CLIP features using proxy queries, which also serve as an interface for multi-level interaction between text and visual modalities. Extensive experiments validate the effectiveness of our method, experimental results on five standard benchmarks have shown that DEDOS achieves state-of-the-art performance.

![Framework](static/images/framework.png)

## Visual Results
![Framework](static/images/qualitatives.png)

## Environment

- Python (3.8.19)
- PyTorch (1.13.1) 
- TorchVision (0.14.1)
- diffusers (0.30.2)
- detectron2 (0.6c)
- transformers (4.44.2)

## Installation

```
conda create -n dedos python=3.8
conda activate dedos
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Dataset Preparation

COCO-Stuff Dataset

- Get COCO 2017 Images:

  ```
  wget http://images.cocodataset.org/zips/train2017.zip
  wget http://images.cocodataset.org/zips/val2017.zip
  ```

- Get COCO-Stuff Annotations:

  Bash

  ```
  wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
  ```

- Unzip: Extract all three downloaded archives.

  - Place the image contents (`train2017` and `val2017` folders) inside `coco-stuff/images/`.
  - Place the annotation contents (segmentation maps) inside `coco-stuff/annotations/`.

- Pre-processing (Generate Labels):

  

  ```
  python datasets/prepare_coco_stuff.py
  ```

  This script generates the final, Detectron2-compatible labels in `annotations_detectron2/`.

ADE20K Datasets (150 and 847 Classes)

- Get ADEChallengeData2016:

  ```
  wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
  ```

- Unzip: Extract the archive. The structure should automatically align with the paths above

- Pre-processing (Generate Labels):

  ```
  python datasets/prepare_ade20k_150.py
  ```

- Get ADE20k-Full: Download the data of ADE20k-Full from https://groups.csail.mit.edu/vision/datasets/ADE20K/request_data/.

  Unzip: Extract the archive. Ensure the core structure is as listed above.

- Pre-processing (Generate Labels):

  ```
  python datasets/prepare_ade20k_847.py
  ```

PASCAL VOC & PASCAL Context Datasets

- Get VOC 2012 Data:

  ```
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  ```

- Get Segmentation Augmentation:

  Bash

  ```
  wget https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip
  ```

- Unzip: Extract both archives. Ensure the contents are placed inside the VOCdevkit/VOC2012/ folder.

- Pre-processing (Generate Labels):

  ```
  python datasets/prepare_voc.py
  ```

- Get VOC 2010 Images:

  ```
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
  ```

- Get Context Annotations (Merged JSON):

  ```
  wget https://codalabuser.blob.core.windows.net/public/trainval_merged.json
  ```

- Get Context Annotations (Segmentation Maps):

  ```
  wget https://roozbehm.info/pascal-context/trainval.tar.gz
  ```

- Unzip: Extract the files and place them inside the VOCdevkit/VOC2010/ folder.

- Pre-processing (Generate Labels):

  ```
  # For 59 classes
  python datasets/prepare_pascal_context_59.py
  
  # For 459 classes
  python datasets/prepare_pascal_context_459.py
  ```



## Evaluation

    python val_net.py \
        [CONFIG_FILE_PATH] \
        [NUM_GPUS] \
        [OUTPUT_DIRECTORY] \
        [FLAG_FOR_EVALUATION] \
        [FLAG_FOR_RESUME] \
        --opts \
            MODEL.WEIGHTS [PATH_TO_MODEL_WEIGHTS] \
            DATASETS.TEST [DATASET_NAME_TUPLE] \
            # ... Other Model or Dataset specific settings ...
            

For example:

```
python val_net.py \
--config-file configs/diff.yaml \
--dist-url auto \
--eval-only \
--machine-rank 0 \
--num-gpus 1 \
--num-machines 1 \
--resume False \
--opts \
    OUTPUT_DIR output/eval \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON datasets/ade150.json \
    DATASETS.TEST "(\"ade20k_150_test_sem_seg\",)" \
    TEST.SLIDING_WINDOW True \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
    MODEL.WEIGHTS DEDOS/train/x0.01-b8-2024-10-23-16:16:04/model_0001.pth

```


    
## Demo
'''
python demo.py \
    --config-file ../configs/diff.yaml \
    --input /datanvme/lf/data/cityscapes_semantic_d2/image/val/lindau_000056_000019_leftImg8bit.png \
    --output /datanvme/lf/debug \
    --opts MODEL.WEIGHTS /datanvme/lf/output/cat-seg/train/debug/2024-10-18-23:48:52/model_final.pth

## Training 

    python train_net.py \
        [CONFIG_FILE_PATH] \
        [NUM_GPUS] \
        [OUTPUT_DIRECTORY_SPECIFIER] \
        [RESUME_FLAG] \
        --opts

For example:

```
python train_net.py \
--config-file configs/vitl_336.yaml \
--dist-url auto \
--num-gpus 1 \
--num-machines 1 \
--machine-rank 0 \
--resume \
--opts \
    OUTPUT_DIR /datanvme0/lifan/output/cat-seg/train/debug/$(date "+%Y-%m-%d-%H:%M:%S")
```


## Acknowledgements

This repo is built upon these previous works:

- [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg)
- [Zegformer](https://github.com/dingjiansw101/ZegFormer)
- [ODISE](https://github.com/NVlabs/ODISE/tree/main)

## Citation

If you find it helpful, you can cite our paper in your work.

    @InProceedings{li2025images,
        author    = {Li, Fan and Wang, Xuanbin and Wang, Xuan and Zhang, Zhaoxiang and Xu, Yuelei},
        title     = {Images as Noisy Labels: Unleashing the Potential of the Diffusion Model for Open-Vocabulary Semantic Segmentation},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        month     = {October},
        year      = {2025},
        pages     = {24255-24265}
    }
