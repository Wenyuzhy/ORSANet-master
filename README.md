# ORSANet: Rethinking Occlusion in FER: A Semantic-Aware Perspective and Go Beyond
<img src="figure\ORSANet.png"  height=400 width=900>

Facial expression recognition (FER) is a challenging task due to pervasive occlusion and dataset biases. In response, we present ORSANet, which introduces the following three key contributions: First, we introduce auxiliary multi-modal semantic guidance to disambiguate facial occlusion and learn high-level semantic knowledge, which is two-fold: 1) we introduce semantic segmentation maps as dense semantics prior to generate semantics-enhanced facial representations; 2) we introduce facial landmarks as sparse geometric prior to mitigate intrinsic noises in FER, such as identity and gender biases. Second, to facilitate the effective incorporation of these two multi-modal priors, we customize a Multi-scale Cross-interaction Module (MCM) to adaptively fuse the landmark feature and semantics-enhanced representations within different scales. Third, we design a Dynamic Adversarial Repulsion Enhancement Loss (DARELoss) that dynamically adjusts the margins of ambiguous classes, further enhancing the model's ability to distinguish similar expressions. We further construct the first occlusion-oriented FER dataset to facilitate specialized robustness analysis on various real-world occlusion conditions, dubbed Occlu-FER.

### Preparation
- Preparing Data:
  Download [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset), [AffectNet](https://mohammadmahoor.com/pages/databases/affectnet/), [RAF-DB_valid_occlu and Occlu-FER](https://wenyuzhy.github.io/Occlu-FER/).
  As an example, assume we wish to run RAF-DB. We need to make sure it have a structure like following:

	```
	- data/raf-db/
		 train/
		     train_00001_aligned.jpg
		     train_00002_aligned.jpg
		     ...
		 valid/
		     test_0001_aligned.jpg
		     test_0002_aligned.jpg
		     ...
	```

- Pretrained model weights:
  Image backbone & Landmark detector from [here](https://drive.google.com/drive/folders/1X9pE-NmyRwvBGpVzJOEvLqRPRfk_Siwq).
  Segmentation Network from [here](https://github.com/Kartik-3004/SegFace) (Swin_Base	224	LaPa). 
  Put entire `pretrain` folder under `models` folder.

	```
	- models/pretrain/
		 ir50.pth
		 mobilefacenet_model_best.pth.tar
  		 model_299.pt
		     ...
	```

### Testing

You can evaluate our model on each dataset by running: 

```
python test.py --checkpoint checkpoint/best.pth -p
```

### Training
Train on RAF-DB dataset:
```
python train.py --gpu 0 --batch_size 20
```
You may adjust batch_size based on your # of GPUs. We provide the log in  `log` folder. You may run several times to get the best results. 




