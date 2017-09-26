# Carvana Semantic Segmentation - solution

[[Challenge link](https://www.kaggle.com/c/carvana-image-masking-challenge)]

This solution gets 0.9945 dice, which is within 0.25% from current top, but still places in the middle of the leaderboard (competition requires pixel-perfect segmentation). 

This is mainly for-fun solution, without any special tricks (no ensembling, no data augmentation), with very little hyperparameter tuning.

Architecture: slightly customized UNet (so that it can process images in native 1918x1280 resolution)

Framework: PyTorch

## Running

1. Extract competition data in `input/`
2. Preprocess data with `python dataset.py` (20 min)
3. Train to convergence with `python train.py` (12+ hours)
4. If necessary, generate submission with `python make_submission.py`

## License
MIT