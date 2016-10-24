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
https://dl.dropboxusercontent.com/u/10728218/data/storylines_data.zip (9.4 GB)

This contains albums with image URLs and fc7 features in the format used by the code.

## Citation

Please cite the following if it helps your research!

    @article{sigurdsson2016learning,
      author = {Gunnar A. Sigurdsson and Xinlei Chen and Abhinav Gupta},
      title = {Learning Visual Storylines with Skipping Recurrent Neural Networks},
      journal = {ArXiv e-prints},
      eprint = {1604.04279}, 
      year = {2016},
      url = {http://arxiv.org/abs/1604.04279},
      pdf = {http://arxiv.org/pdf/1604.04279.pdf}
    }
