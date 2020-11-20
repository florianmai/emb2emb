# Introduction

This repository contains the code for the paper 
[Plug and Play Autoencoders for Conditional Text Generation](https://arxiv.org/abs/2010.02983)
by Florian Mai, Nikolaos Pappas, Ivan Montero, Noah A. Smith, and James Henderson published at EMNLP 2020.

The paper proposes a framework that allows to use pretrained text autoencoders for conditional text
generation tasks such as style transfer.

It consists of three stages as shown below. In the pretraining stage, a text autoencoder is trained on unlabeled data.
In the task training stage, a mapping is trained that maps the embedding of the input to the embedding of the output.
In the inference stage, the encoder, mapping, and decoder are combined to solve the task.


![Emb2Emb Training Framework](images/pnpframework.png)

# Requirements

The code was tested under Ubuntu 18.04 and Debian 10, Python3.7 and PyTorch 1.7.

The required Python packages can be installed via
```bash
pip install -r requirements.txt
```

# Code Structure

The code consists of the `autoencoders` package, which handles the pretraining of an
RNN autoencoder, and the `emb2emb` package, which handles the task training and inference
stages. The root folder brings these together and applies the framework to Yelp sentiment transfer
and WikiLarge sentence simplification.

* train_autoencoder.py: Handles the autoencoder pretraining
* train.py: Handles Emb2Emb training.
* data.py: Manages loading of datasets and certain evaluation functions.
* emb2emb_autoencoder.py: Wraps an autoencoder with the interface required by Emb2Emb
* classifier.py: Trains a binary classifier on text data (e.g. for sentiment transfer).

## emb2emb

Implements task training and inference stages.

* trainer.py: Implements the workflow of Emb2Emb, including the adversarial term.
* encoding.py: Defines encoder and decoder interfaces to implement so they work with Emb2Emb
* fgim.py: Implements FGIM
* losses.py: Defines losses as in the paper, except the adversarial loss.
* mapping.py: Implements mappings, such as OffsetNet and MLP.

## autoencoders
Implements autoencoder pretraining.

* rnn_encoder.py: Implements an RNN-based encoder.
* rnn_decoder.py: Implements an RNN-based decoder.


* autoencoder.py: Besides encoding and decoding, manages regularization methods like adding noise to the input or special training losses, such as VAEs or adversarial autoencoders.
* data_loaders.py: Preprocesses text data into HDF5 files for quicker training afterwards.
* noise.py: Computes noise for denoising autoencoders.

# Example: Sentence Simplification on WikiLarge

(Further experiments can be found in the `experiments` folder)

There are two steps: We first need to pretrain the autoencoder on the text data without labels, and then
train Emb2Emb on the data with labels.

## Pretrain autoencoder

### Prepare data for autoencoder training.
First, you need to preprocess your data and bring it into the HDF5 file format expected by the autoencoder training script.
To this end, process your file 'all_train', which contains all texts available at training time, 1 sentence per line,
via the following command:

```bash
cd autoencoders
python data_loaders.py ../data/wikilarge/all_train ../wikilarge/wiki.h5 64 -t CharBPETokenizer -mw 30000
```

### Build config file
Autoencoder training is configured through a config file, for which autoencoders/config/default.json is a good template.

### Train
After configuring the config file, you can train the autoencoder with the following command:
```bash
python train_autoencoder.py experiments/wikilarge/dae_config.json
```


## Train Emb2Emb on WikiLarge

Train OffsetNet with adversarial regularization on WikiLarge.

```bash
python train.py --embedding_dim 1024 --batch_size 64 --lr 0.0001 --modeldir ./tmp/wiki-ae/lstmae0.0p010/ --data_fraction 1.0 --n_epochs 10 --n_layers 1 --print_outputs --dataset_path data/wikilarge --validate --mapping offsetnet --hidden_layer_size 1024 --loss cosine --adversarial_regularization --adversarial_lambda 0.032 --outputdir tmp/wikilarge/ --binary_classifier_path no_eval --output_file emnlp_wiki_offset_cosine.csv --real_data_path input --max_prints 20 --device cpu
```

# Citation and Acknowledgement

If you find our work useful, or use our code, please cite us as:

```bibtex
@inproceedings{mai-etal-2020-plug,
    title = "Plug and Play Autoencoders for Conditional Text Generation",
    author = "Mai, Florian  and
      Pappas, Nikolaos  and
      Montero, Ivan  and
      Smith, Noah A.  and
      Henderson, James",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.491",
    pages = "6076--6092",
    abstract = "Text autoencoders are commonly used for conditional generation tasks such as style transfer. We propose methods which are plug and play, where any pretrained autoencoder can be used, and only require learning a mapping within the autoencoder{'}s embedding space, training embedding-to-embedding (Emb2Emb). This reduces the need for labeled training data for the task and makes the training procedure more efficient. Crucial to the success of this method is a loss term for keeping the mapped embedding on the manifold of the autoencoder and a mapping which is trained to navigate the manifold by learning offset vectors. Evaluations on style transfer tasks both with and without sequence-to-sequence supervision show that our method performs better than or comparable to strong baselines while being up to four times faster.",
}
```

Thanks to [Ivan Montero](https://github.com/ivanmontero), who was responsible for
implementing most of the autoencoder pre-training. He is an awesome student who you should look out for hiring in the future! 
