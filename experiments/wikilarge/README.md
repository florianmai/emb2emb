# Experiments on WikiLarge

The experiments on WikiLarge sentence simplification require two steps: First, train an 
autoencoder model. Second, the supervised sentence simplification model is trained.

## Train Autoencoder

Create a folder for saving models, etc.:

```bash
mkdir tmp
mkdir tmp/wikilarge/
```

First, prepare the data for training the autoencoder:
```bash
cd autoencoders
python data_loaders.py ../data/wikilarge/all_train ../tmp/wikilarge/wiki.h5 64 -t CharBPETokenizer -mw 30000
```

The following command trains an autoencoder as configured in the respective *.json file.
```bash
python train_autoencoder.py experiments/wikilarge/dae_config.json
```

## Comparison of Network Architectures

To run sentence simplification on WikiLarge with different mappings, use the following 
command with `<mapping>` set to `offsetnet`, `mlp`, or `resnet`, respectively.

```bash
python train.py --embedding_dim 1024 --batch_size 64 --lr 0.0001 --modeldir ./tmp/wikilarge/lstmae0.0p010/ --data_fraction 1.0 --n_epochs 10 --n_layers 1 --print_outputs --dataset_path data/wikilarge --validate --mapping <mapping> --hidden_layer_size 1024 --loss cosine --adversarial_regularization --adversarial_lambda 0.032 --outputdir tmp/wikilarge/ --binary_classifier_path no_eval --output_file emnlp_wiki_mappings_cosine.csv --real_data_path input --max_prints 20 --device cpu --validation_frequency -1 --log_freq 100
```

## End-to-End Seq2Seq Training

Run the following command to compute the S2S-Freeze baseline from the paper.

```bash
python train.py --embedding_dim 1024 --batch_size 64 --lr 0.00005 --modeldir ./tmp/wikilarge/lstmae0.0p010/ --data_fraction 1.0 --n_epochs 20 --n_layers 1 --print_outputs --dataset_path data/wikilarge --validate --mapping mlp --hidden_layer_size 1024 --loss ce --mode seq2seq_freeze --binary_classifier_path no_eval --output_file emnlp_wiki_end2end_freeze.csv --outputdir ./tmp/wikilarge/
```