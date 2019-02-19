# CNN-for-Sentence-Classification
CNN for Sentence Classification

THis is an implementation with some modifications of the paper [Convolutional Neural Networks for Sentence Classification(2014 Kim et al.)](http://www.emnlp2014.org/papers/pdf/EMNLP2014181.pdf). 

We use pre-trained word embeddings to train the model, which is [wiki-news-300d-1M.vec.zip](http://fasttext.cc/docs/en/english-vectors.html) of fastText, an open-sourced library for efficient text classification and representation learning by facebook. Before running the code, you should first download and put it under the ```word_embedding``` directory. The dataset used can be found at [CS 11-747 Assignment 1 Topic Classification data](http://www.phontron.com/class/nn4nlp2019/assignments.html). You should also download and put it under the ```dataset``` directory.

Using the parameter configuration in the code, the best accuracy achieved on the validation set is 84.11%.

To run the code, you should specify the random seed for numpy and torch module to induce reproducibility, as follows.
```
python train_valid_test.py --seed 1234
```
