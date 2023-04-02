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
| Ours(ResNet50) | 23.6 | 3.78 | 0.695 | 0.903 | 0.769 | - | - | - | - | - | - | - |
  - AP is calculated on `flip_test=True`

### Look into more: lightweight backbones
The backbones used in the paper are ResNet50 and HRNet which are not suitable on mobile devices. There are some tests applying lightweight backbones on this model. The backbones are like the below.
  - Basically `MoibleNetV2`, which is the worldwide-used backbone network.
  - `MobileNetV3-Large`, which is comparable with MobileNetV2.
  - `GhostNetV2`, which has more params but, more efficient than any other backbones.

| Model | #Params<br>(M) | GFLOPs | AP | model size(MB) | ms | memory access |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| Ours<br>(MobileNetV2) | 2.31 | 0.29 | 0.598 | ... | ... | ... |
| Ours<br>(MobileNetV3Large) | 2.31 | 0.29 | 0.598 | ... | ... | ... |
| Ours<br>(GhostNetV2) | 2.31 | 0.29 | 0.598 | ... | ... | ... |
  - `AP` is calcualted `flip=False`, because the condition `flip` is not appropriate to mobile env.
  - all tests on `iPhone XS`

### Look into more: small inputs
| Model | input size | #Params<br>(M) | GFLOPs | AP | AP.5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| Ours(ResNet50) | 128x96 | 23.6 | 3.78 | 0.695 | 0.903 | 0.769 | - | - | - | - | - | - | - |



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
- [x] hydra 대신 일반 yaml 파일로 학습 파이프라인 수정
- [x] tfrecord 만드는 코드 추가
- [x] 현재 모델들의 #params, GFLOPs 확인하여 기록
- [ ] 모델이 고정되었을때 성능을 높일 수 있는 방법 모색
  - [x] albumentation
  - [ ] different optimizers
  - [ ] ema
- [ ] 동일한 성능 재현을 위한 seed 고정 방법 찾아보기
- [ ] 모바일 성능 비교할 벤치마크 모델 선정하기


## References
- [Jeff-sjtu/res-loglikelihood-regression](https://github.com/Jeff-sjtu/res-loglikelihood-regression)
