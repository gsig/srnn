**Update 8/3/2018**: We have added a PyTorch implementation of an SRNN layer that is a drop in replacement for nn.LSTM. This layer has been minimally tested and is under development.

Matlab implementation for Skipping Recurrent Neural Networks from:

> **_Learning Visual Storylines with Skipping Recurrent Neural Networks_** <br /> 
Gunnar A. Sigurdsson, Xinlei Chen, Abhinav Gupta <br /> 
http://arxiv.org/abs/1604.04279
  
The code is a MATLAB implementation of a Recurrent Neural Network, wrapped by the S-RNN architecture. This code was recently released, so please let me know if you encounter any strange behaviour.

The code is organized as follows:
- ```rnn_trainnet.m``` is a script used for training the S-RNN
- ```rnn_testnet.m``` is a script used for generating a summary of multiple albums, and selecting the best summary
- ```predictnext_short.m``` is a code to run the short-term prediction experiment from the paper
- ```predictnext_long.m``` is a code to run the long-term prediction experiment from the paper



## Data
The subset of Yahoo Flickr 100M used in the paper:
https://www.dropbox.com/s/ghrieaikaobukz4/storylines_data.zip?dl=1 (9.4 GB)

This contains albums with image URLs and fc7 features in the format used by the code.

## Data for evaluating long-term prediction:
https://www.dropbox.com/s/66cal9wa0iumgoa/evaluation.zip?dl=1 (0.4 GB)

Using the below models should result in the following numbers corresponding to columns in Fig 6. right side in the paper.
- srnn: 0.310648
- rand: 0.214815
- nn: 0.2875

## Pre-trained models
Pre-trained models:
https://www.dropbox.com/s/zpuvm436ortrsw6/srnnmodels.zip?dl=1 (0.02 GB)

## Citation

Please cite the following if it helps your research!

    @article{sigurdsson2016learning,
      author = {Gunnar A. Sigurdsson and Xinlei Chen and Abhinav Gupta},
      title = {Learning Visual Storylines with Skipping Recurrent Neural Networks},
      journal = {European Conference on Computer Vision},
      year = {2016},
      url = {http://arxiv.org/abs/1604.04279},
      poster = {http://www.eccv2016.org/files/posters/P-3A-26.pdf},
      pdf = {http://arxiv.org/pdf/1604.04279.pdf}
    }
