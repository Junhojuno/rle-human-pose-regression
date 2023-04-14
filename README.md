# Human Pose Regression with Tensorflow
Human Pose Regression(HPR) is simple to estimate keypoints of human since it does not have any postprocess that transforms heatmaps to coordinates.
HPR has a drawback that its accuracy is much lower than that of heatmap-based models. but recently, with flow-based model, HPR has so improved that it can be worth replace heatmap-based model.

> [Human Pose Regression with Residual Log-likelihood Estimation](https://arxiv.org/abs/2107.11291) <br>
> Jiefeng Li, Siyuan Bian, Ailing Zeng, Can Wang, Bo Pang, Wentao Liu, Cewu Lu <br>
> ICCV 2021 Oral

<br>

this repo refered to the below research and its official repo, and looked into more about practicality.

## Results

### COCO Validation Set
To compare with the official results, regression model(Tensorflow) has trained on MSCOCO and the official configuration.

| Model | #Params<br>(M) | GFLOPs | AP | AP.5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| Benchmark<br>(ResNet50) | 23.6 | 4.0 | 0.713 | 0.889 | 0.783 | - | - | - | - | - | - | - |
| Ours(ResNet50) | 23.6 | 3.78 | 0.694 | 0.904 | 0.760 | 0.668 | 0.736 | 0.727 | 0.912 | 0.786 | 0.695 | 0.776 |
  - AP is calculated on `flip_test=True`

### Look into more: lightweight backbones
The backbones used in the paper are ResNet50 and HRNet which are not suitable on mobile devices. There are some tests applying lightweight backbones on this model. The backbones are like the below.
  - Basically `MoibleNetV2`, which is the worldwide-used backbone network.
  - `EfficientNet-B0`, which has a considerable score with fast inference.
  - `GhostNetV2`, which has more params but, more efficient than any other backbones.

| Model | #Params<br>(M) | GFLOPs | AP | model size<br>(MB) | latency<br>(ms) | memory access |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| Ours<br>(MobileNetV2)     | 2.31 | 0.2935 | 0.598 | ... | ... | ... |
| Ours<br>(EfficientNet-B0) | 4.09 | 0.3854 | ????? | ... | ... | ... |
| Ours<br>(GhostNetV2 1.0x) | 3.71 | 0.1647 | ????? | ... | ... | ... |
  - `AP` is calcualted `flip=False`, because the `flip` inference is not used on mobile.
  - mobile test based on iOS(`iPhone XS`)

### Look into more: small inputs
| Model | input size | #Params<br>(M) | GFLOPs | AP | AP.5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| Ours(ResNet50) | 128x96 | 23.6 | 3.78 | 0.694 | 0.904 | 0.760 | 0.668 | 0.736 | 0.727 | 0.912 | 0.786 | 0.695 | 0.776 |


<br>

## Setup

### environment
Before starting, `docker`, `nvidia-docker` should be set.  
  - tensorflow==2.9.3
  - tensorflow-addons==0.19.0
  - tensorflow_probability==0.17.0
```bash
docker build -t rle:tf .
```

### project tree
Before cloning this repo, you have to set the dir tree like below. if not, the codes all will throw errors.
```bash
root
├── datasets
│   └── mscoco
│        ├── annotations
│        └── images
├── $project_dir
│   ├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── README
│   └── ...
└── ...
``` 

### data
Train & evaluation are operated on tfrecord files. so download the raw dataset form https://cocodataset.org and convert it to `.tfrecord`. <br>
According to the dir tree mentioned above, it is easy to convert, just run the code below. If not following the tree, should change the current dir using `-c` option on command line.
```python
python write_tfrecord.py
```

### training
```python
python train.py -c config/256x192_res50_regress-flow.yaml
```

## TO-DO
- [ ] 다양한 backbone에 대해 학습 진행
- [ ] 모바일(iOS) 배포. 


## References
- [Jeff-sjtu/res-loglikelihood-regression](https://github.com/Jeff-sjtu/res-loglikelihood-regression)
