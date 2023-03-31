# Human Pose Regression with Residual Log-likelihood Estimation with Tensorflow

> [Human Pose Regression with Residual Log-likelihood Estimation](https://arxiv.org/abs/2107.11291) <br>
> Jiefeng Li, Siyuan Bian, Ailing Zeng, Can Wang, Bo Pang, Wentao Liu, Cewu Lu <br>
> ICCV 2021 Oral

According to the official code, this repo is created for rewriting it with Tensorflow, at the same time, check the model efficiency on mobile.

## Results

### COCO
| Model | #Params(M) | GFLOPs | AP | AP50 | AP75 |
| :------------- | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| Benchmark<br>(ResNet50) | 23.6 | 4.0 | 0.713 | 0.889 | 0.783 |
| ResNet50 | 23.6 | 3.78 | 0.695 | 0.903 | 0.769 |
| ResNet50 (w/ album) | 23.6 | 3.78 | 0.689 | 0.904 | 0.761 |
- AP is calculated on `flip_test=True`
- I have a quite different result from the origianl's when training just as it is in the official repo. So, various options are considered.

## Lightweight Backbones
The official did not care on running on mobile.

| Model | #Params(M) | GFLOPs | AP | model size(MB) | ms | memory access |
| :------------- | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| Benchmark<br>(ResNet-50) | 23.6 | 4.0 | 0.713 | ... | ... | ... |
| MobileNetV2 (w/o flip) | 2.31 | 0.29 | 0.598 | ... | ... | ... |
| MobileNetV2 | 2.31 | 0.29 | 0.613 | ... | ... | ... |
- All models are tested on `iPhone XS`
- AP is calculated on `flip_test=True`
- add a mobile benchmark model to compare

## Usage

### Environment
After downloading `docker` and `nvidia-docker` and clone this repo, just run the below script on the cmd at current dir. <br>
Feel free to name the image.
  - tensorflow==2.11
  - tensorflow-addons==0.19.0
  - tensorflow_probability==0.19.0
```bash
docker build -t rle:tf .
```

### Before starting
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

### Convert to TFRecord
Train & evaluation are operated on tfrecord files. so download the raw dataset form https://cocodataset.org and convert it to `.tfrecord`. <br>
According to the dir tree mentioned above, it is easy to convert, just run the code below. If not following the tree, should change the current dir using `-c` option on command line.
```python
python write_tfrecord.py
```

### Train / Evaluation
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
