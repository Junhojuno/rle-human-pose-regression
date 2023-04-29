# Human Pose Regression with Tensorflow

Human Pose Regression(HPR) is simple to estimate keypoints of human since it does not have any postprocess that transforms heatmaps to coordinates.
HPR has a drawback that its accuracy is much lower than that of heatmap-based models. but recently, with flow-based model, HPR has so improved that it can be worth replace heatmap-based model.

> [Human Pose Regression with Residual Log-likelihood Estimation](https://arxiv.org/abs/2107.11291) <br>
> Jiefeng Li, Siyuan Bian, Ailing Zeng, Can Wang, Bo Pang, Wentao Liu, Cewu Lu <br>
> ICCV 2021 Oral

<p align="center">
  <img src="https://user-images.githubusercontent.com/38845662/234257766-da4d0cb7-908e-4fe0-84ee-dabfbd2c17ec.gif">
  <img src="https://user-images.githubusercontent.com/38845662/234259627-cc495d62-2682-4ff2-a4e1-9eb63d887b2d.gif">
</p>

<br>

Looking into [the officials](https://github.com/Jeff-sjtu/res-loglikelihood-regression), there are only basic sources for reproducing scores written on the paper. Ummm...those are also important but practical experiments should be executed, such as test with mobile backbone, mobile deployment, ... etc. Let's have these!

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

After training, something noticable is that there is a small amount of difference between `flip=true` and `flip=false`, which is much lower than that of heatmap-based models.

| Model | #Params<br>(M) | GFLOPs | AP | model size<br>(MB) | latency<br>(ms) | memory access |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| Ours<br>(MobileNetV2)     | 2.31 | 0.2935 | 0.600 | 4.7 | 10~11 | ... |
| Ours<br>(EfficientNet-B0) | 4.09 | 0.3854 | 0.665 | 8.3 | 5~6 | ... |
| Ours<br>(GhostNetV2 1.0x) | 3.71 | 0.1647 | 0.624 | 7.6 | 9~10 | ... |
  - `AP` is calcualted `flip=False`, because the `flip` inference is not used on mobile.
  - The model is tested on Android device.

### Look into more: small inputs
| Model | input size | #Params<br>(M) | GFLOPs | AP | AP.5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| Ours(ResNet50) | 128x96 | 23.6 | 3.78 | - | - | - | - | - | - | - | - | - | - |


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
- [ ] low-resolution에서의 모델 성능 측정
- [ ] 모바일 


## References
- [Jeff-sjtu/res-loglikelihood-regression](https://github.com/Jeff-sjtu/res-loglikelihood-regression)
