

# Experiments on Yelp


The experiments on Yelp sentiment transfer require three steps: First, train an 
autoencoder model. Second, train a sentiment classifier on top of the autoencoder's
embedding space. Last, the sentiment transfer model is trained.

## Train Autoencoder

Create a folder for saving models, etc.:

```bash
mkdir tmp
mkdir tmp/yelp/
```

First, prepare the data for training the autoencoder:
```bash
cd autoencoders
python data_loaders.py ../data/yelp/all_train ../tmp/bpeyelp/yelp.h5 64 -t CharBPETokenizer -mw 30000
```

The following command trains an autoencoder as configured in the respective *.json file.
```bash
python train_autoencoder.py experiments/yelp/p10.json
```

## Train Sentiment Classifier

The command
```bash
python train.py --embedding_dim 512 --batch_size 64 --lr 0.00005 --modeldir ./tmp/yelp/p10/ --data_fraction 1.0 --n_layers 1 --print_outputs --dataset_path data/yelp --validate --mapping identity --hidden_layer_size 512 --loss fliploss --baseloss cosine --outputdir ./tmp/yelp/ --n_epochs_binary 10 --lr_bclf 0.0001 --unaligned --outputmodelname daeyelp_noise50_dropout50 --dropout_binary 0.5 --max_prints 20 --output_file emnlp_yelp_FGIM.csv --binary_classifier_path <binary-classifier-path> --n_epochs 0
```
trains a sentiment classifier and saves it to `./tmp/daeyelp_noise50_dropout50`, so it can be reused in other experiments.

### OffsetNet

Train OffsetNet for sentiment transfer with the following command. 
To train models with increasingly greater emphasis on changing the style/sentiment, 
replace `lambda-clf` with increasingly larger values (0.1, 0.5, 0.9, 0.95, 0.99).
If `<binary-classifier-path>` is specified, it must point to a model that is 
compatible with the [transformers](https://huggingface.co/transformers/) library 
`AutoModelForSequenceClassification` class and was trained on binary sentiment prediction.
If the path is not specified, the same classifier from above that is used for 
guiding the model is used for evaluation, too. 
This should only be used for the purpose of testing code, because it will give drastically overconfident results.
```bash
python train.py --embedding_dim 512 --batch_size 64 --lr 0.00005 --modeldir ./tmp/yelp/p10/ --data_fraction 1.0 --n_layers 1 --print_outputs --dataset_path data/yelp --validate --mapping offsetnet --hidden_layer_size 512 --loss fliploss --baseloss cosine --adversarial_regularization --adversarial_lambda 0.008 --outputdir ./tmp/yelp/ --n_epochs_binary 10 --lr_bclf 0.0001 --unaligned --outputmodelname daeyelp_noise50_dropout50 --dropout_binary 0.5 --max_prints 20 --load_binary_clf --output_file emnlp_yelp_offsetnet_sim_20epochs.csv --binary_classifier_path <binary-classifier-path> --real_data_path ./data/yelp/all_train --lambda_clfloss <lambda-clf>
```

### OffsetNet + FGIM

For running the previously trained OffsetNet models with FGIM at inference time, run the following command by plugging in the previously `path-to-emb2emb-model>` (from above) alongside increasing `t` values (0.5, 0.1, 0.01, 0.001, 0.0001).

```bash
python train.py --embedding_dim 512 --batch_size 64 --lr 0. --modeldir ./tmp/yelp/p10/ --n_epochs 0 --data_fraction 1.0 --n_layers 1 --print_outputs --dataset_path data/yelp --validate --mapping offsetnet --hidden_layer_size 512 --loss fliploss --baseloss cosine --adversarial_regularization --outputdir ./tmp/yelp/ --n_epochs_binary 10 --lr_bclf 0.0001 --unaligned --outputmodelname daeyelp_noise50_dropout50 --dropout_binary 0.5 --max_prints 20 --load_binary_clf --output_file emnlp_yelp_offsetnet_sim_advfgim.csv --binary_classifier_path <binary-classifier-path> --real_data_path ./data/yelp/all_train --fast_gradient_iterative_modification --fgim_use_training_loss --load_emb2emb_path <path-to-emb2emb-model> --fgim_threshold <t> --lambda_clfloss 0.5 --adversarial_lambda 0.008
```

### FGIM

The command for running plain FGIM is the following. Replace <t> with different values (0.5, 0.1, 0.01, 0.001, 0.0001) to obtain different levels of sentiment transfer.

```bash
python train.py --embedding_dim 512 --batch_size 64 --lr 0.00005 --modeldir ./tmp/yelp/p10/ --data_fraction 1.0 --n_layers 1 --print_outputs --dataset_path data/yelp --validate --mapping identity --hidden_layer_size 512 --loss fliploss --baseloss cosine --outputdir ./tmp/yelp --n_epochs_binary 10 --lr_bclf 0.0001 --unaligned --outputmodelname daeyelp_noise50_dropout50 --dropout_binary 0.5 --max_prints 20 --load_binary_clf --output_file emnlp_yelp_FGIM.csv --binary_classifier_path <binary_classifier_path> --n_epochs 0 --fast_gradient_iterative_modification  --fgim_threshold <t>
```

### Shen et al. (2020) baseline

The Shen et al. (2020) baseline can be computed with the following command. To obtain different rates of sentiment transfer accuracy, vary <factor> (1.0, 1.5, 2.0, 2.5, 3.0).

```bash
python train.py --embedding_dim 512 --batch_size 64 --lr 0.00000 --modeldir ./tmp/yelp/p10/ --data_fraction 1.0 --n_layers 1 --print_outputs --dataset_path data/yelp --validate --hidden_layer_size 512 --loss fliploss --n_epochs 1 --baseloss cosine --outputdir ./tmp/yelp/ --n_epochs_binary 10 --lr_bclf 0.0001 --unaligned --outputmodelname daeyelp_noise50_dropout50 --dropout_binary 0.5 --max_prints 20 --load_binary_clf --output_file emnlp_yelp_shen2019.csv --binary_classifier_path <binary-classifier-path> --real_data_path ./data/yelp/all_train --meanoffsetvector_factor <factor> --mapping meanoffsetvector
```