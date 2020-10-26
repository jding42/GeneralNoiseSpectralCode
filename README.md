# GeneralNoiseSpectralCode

This repo contains code for our paper https://arxiv.org/abs/2008.13735  [**Quantum Entropy Scoring for Fast Robust Mean Estimation and Improved Outlier Detection**]


[Jingqiu Ding](Jingqiu.Ding@inf.ethz.ch), [Sam Hopkins](http://www.samuelbhopkins.com/), [David Steurer](https://www.dsteurer.org)

## Code structure
* [`Main.py`](Main.py) contains the main file to run for testing different algorithms on different models.
* [`Model.py`](Model.py) generate various spiked matrix models used in the experiment
*  [`truncationPCA.py`](truncationPCA.py)[`nbwAlgo.py`](nbwAlgo.py)[`sawAlgo.py`](sawAlgo.py): algorithms by truncation PCA, non-backtracking walk estimator, and self-avoiding walk estimator

*  [`dataProcess.py`](dataProcess.py): the file to run for data processing and plotting
