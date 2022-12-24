# Human Pose Regression Baseline
The main goal of this repository is to rewrite the human pose regression with a Tensorflow and better structure for better portability and adaptability to apply new experimental methods. The human pose regression pipeline is based on [this paper](https://arxiv.org/abs/2107.11291).

## Performance
| model | # Params | FLOPs | AP | AP50 | AP75 | link |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| ResNet-50 | Content Cell | Content Cell | Content Cell | Content Cell | Content Cell | Content Cell |

## ✔️ Set environment
- conda env / docker
- Tensorflow 2.9+
- Tensorflow-Probability 0.17.0+

## ✔️ Convert data to tfrecord
- download MSCOCO
- convert to tfrecord

## ✔️ Train / Evaluation
```python
sh train.sh
```

## ✔️ Advanced
- Automatic Mixed-Precision Training
- 


## :thinking: TO-DO
  - [ ] various setting for stable training
  - [ ] Compare HRNet-W32/W48 with ResNet50
  - [ ] Compare MobileNetV2 with ResNet50
  - [ ] demo webcam / video
  - [ ] web and mobile export
  - [ ] exported model evaluation

## ✔️ Pretrained Models


## ✔️ References

