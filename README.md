# Batch Renormalization
Batch Renormalization algorithm implementation in Keras 1.2.1. Original paper by Sergey Ioffe, [Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models](https://arxiv.org/pdf/1702.03275.pdf).

# Usage
Add the `batch_renorm.py` script into your repository, and import the BatchRenormalization layer.

Eg. You can replace Keras BatchNormalization layers with BatchRenormalization layers. 
```
from batch_renorm import BatchRenormalization
```

# Performance
Using BatchRenormalization layers requires slightly more time than the simpler BatchNormalization layer. 

Observed speed differences in WRN-16-4 with respect to BatchNormalization on a 980M GPU:

1) **Batch Normalization** : 137 seconds per epoch.

2) **Batch Renormalization (Mode 0)** : 152 seconds per epoch.

3) **Batch Renormalization (Mode 2)** : 142 seconds per epoch.

# Results
The following graph is from training a Wide Residual Network (WRN-16-4) on the CIFAR 10 dataset, with no data augmentation and no dropout. Therefore all models clearly overfit. 

However, the graphs compare WRN-16-4 model with Keras BatchNormalization (mode 0) with BatchRenormalization (mode 0 and mode 2). All other parameters are kept constant.

![Training curve](https://github.com/titu1994/BatchRenormalization/blob/master/plots/batchnorm_vs_renorm.png?raw=true)

# Requirements
Keras 1.2.1 (will be updated when Keras 2 launches)
Theano / Tensorflow
h5py
seaborn (optional, for plotting training graph)
