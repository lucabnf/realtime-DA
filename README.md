# Real-time domain adaptation in semantic segmentation project
This project addresses the semantic segmentation task in a real-time autonomous driving scenario, dealing with domain adaptation between two different datasets. In particular, training our segmentation model on a huge synthetic dataset, we try to improve its performance on a real-world dataset, exploting an unsupervised adversarial training strategy. In this context, three different discriminator solutions are proposed, along with different optimization and regularization techniques, to both improve performance and reduce training and inference time.  

## Datasets
Download (real-world) CamVid dataset from [Google Drive](https://drive.google.com/file/d/1CKtkLRVU4tGbqLSyFEtJMoZV2ZZ2KDeA/view?usp=sharing) 
Download (synthetic) IDDA dataset from [Google Drive](https://drive.google.com/file/d/1GiUjXp1YBvnJjAf1un07hdHFUrchARa0/view)
  
## Train
# 1) segmentation train (BiSeNet)
python seg_train.py
# 2) adversarial train for unsupervised domain adaptation (IDDA+CamVid)
python adv_train.py

## Test
python eval.py