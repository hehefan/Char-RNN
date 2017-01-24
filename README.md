# Char-RNN
## Basic charater-level language model 
Basic-Char-RNN
## ACT charater-level language model
ACT-Char-RNN: Implement the paper [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/pdf/1603.08983v4.pdf) with TensorFlow. However, I get a reverse phenomenon contrast to the ACT paper. The neural network ponders less on space charecter than others. For example:

| char | remaining probility| iteration times|
|:----:|:------------------:|:--------------:|
|   t  |      0.144139      |        4       |
|   o  |      0.069154      |        7       |
|      |      0.245744      |        2       | 
|   m  |      0.101206      |        4       |
|   o  |      0.149854      |        3       |
|   v  |      0.172533      |        2       |
|   e  |      0.189886      |        3       |
|      |      0.309559      |        2       |
|   t  |      0.016886      |        8       |
|   h  |      0.063482      |        7       |
|   e  |      0.145703      |        5       |
|   i  |      0.445629      |        2       |
|   r  |      0.194783      |        3       |

The code is inspired and adjusted from [DeNeutoy](https://github.com/DeNeutoy/act-tensorflow) and [abhitopia](https://github.com/abhitopia/ACT_Tensorflow).
