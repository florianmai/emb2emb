import os
import sys
import time
import argparse
import random

import numpy as np

import torch
import torch.nn as nn

from torch.nn.modules.loss import MSELoss, CrossEntropyLoss
from emb2emb.losses import CosineLoss, FlipLoss
from classifier import train_binary_classifier
from emb2emb.fgim import binary_classification_criterion,\
    make_binary_classification_loss, not_matched

DEFAULT_CONFIG = "autoencoders/config/default.json"

from emb2emb.mapping import MLP, OffsetNet, MeanOffsetVectorMLP, ResNet
from emb2emb_autoencoder import AEEncoder, AEDecoder
from data import get_data
from emb2emb.trainer import Emb2Emb, MODE_EMB2EMB, MODE_FINETUNEDECODER, MODE_SEQ2SEQ, MODE_SEQ2SEQFREEZE


def get_train_parser():
    parser = argparse.ArgumentParser(description='Emb2Emb')
    # paths
    parser.add_argument("--dataset_path", type=str,
                        required=True, choices=["data/yelp", "data/wikilarge"], help="Path to dataset")
    parser.add_argument("--outputdir", type=str,
                        default='savedir/', help="Output directory")
    parser.add_argument("--outputmodelname", type=str, default='model.pickle')
    parser.add_argument("--modeldir", type=str,
                        default=None, help="Path to autoencoder dir")
    parser.add_argument("--model_name", type=str,
                        default="model.pt", help="Name of the model file.")
    parser.add_argument("--vocab_path", type=str,
                        default="", help="Path to vocabulary.")
    parser.add_argument("--real_data_path", type=str, default="input",
                        help="If 'input' is specified, we use the target sequence embeddings for adversarial regularization. Otherwise randomly sample from the data file given at the path.")
    parser.add_argument("--binary_classifier_path", type=str, default=None,
                        help="Path to the BERT SequenceClassification model and it's tokenizer.")
    parser.add_argument("--output_file", type=str, default='output.csv',
                        help="Output file for csv to store results.")
    parser.add_argument("--load_emb2emb_path", type=str, default=None,
                        help="Path to already trained mapping model.")

    # training
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validation_frequency", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_bclf", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--n_epochs_binary", type=int, default=5)
    parser.add_argument("--load_binary_clf", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mode", type=str, default=MODE_EMB2EMB, help="The training mode to use.",
                        choices=[MODE_EMB2EMB, MODE_FINETUNEDECODER, MODE_SEQ2SEQ, MODE_SEQ2SEQFREEZE])

    # model
    parser.add_argument("--mapping", type=str, default='mlp', help="mapping architecture to use",
                        choices=["resnet", "meanoffsetvector", "mlp", "identity", "offsetnet"])
    parser.add_argument("--dropout_p", type=float, default=0.,
                        help="Amount of dropout to have in the mapping model.")
    parser.add_argument("--dropout_binary", type=float, default=0.,
                        help="Amount of dropout in binary classifier.")
    parser.add_argument("--gaussian_noise_binary", type=float, default=0.,
                        help="Amount of gaussian noise in binary classifier.")
    parser.add_argument("--offset_dropout_p", type=float, default=0.,
                        help="Amount of dropout to have in the offset vectors of OffsetNetworks.")
    parser.add_argument("--meanoffsetvector_factor", type=float,
                        default=2., help="Initialization for MeanOffsetVector factor.")
    parser.add_argument("--loss", type=str, default='cosine',
                        help="loss", choices=["mse", "cosine", "ce", "fliploss"])
    parser.add_argument("--baseloss", type=str, default='cosine', help="loss",
                        choices=["mse", "cosine"])
    parser.add_argument("--lambda_clfloss", type=float, default=0.5,
                        help="Weight of the clf loss in comparison to the baseloss. Specify between 0 and 1.")
    parser.add_argument("--n_layers", type=int, default=1,
                        help="Number of layers to use in the Emb2Emb model.")
    parser.add_argument("--hidden_layer_size", type=int, default=1024,
                        help="Hidden layer size to use in the Emb2Emb model.")
    parser.add_argument("--autoencoder", type=str, default="FromFile",
                        help="Specify the autoencoder to use.", choices=["FromFile", "RAE"])
    parser.add_argument("--fast_gradient_iterative_modification", action="store_true",
                        help="Follow the gradient of the binary classifier to change the label.")
    parser.add_argument("--fgim_decay", type=float, default=1.0)
    parser.add_argument("--fgim_threshold", type=float, default=0.001)
    parser.add_argument("--fgim_use_training_loss", action="store_true")
    parser.add_argument("--fgim_start_at_y", action="store_true")
    parser.add_argument("--fgim_no_stop_criterion", action="store_true")
    parser.add_argument("--fgim_weights", type=float,
                        nargs="+", default=[10e0, 10e1, 10e2, 10e3])

    # adversarial reg for mapping
    parser.add_argument("--adversarial_regularization", action="store_true",
                        help="Perform adversarial regularization while training mapping.")
    parser.add_argument("--critic_lr", type=float,
                        default=0.00001, help="LR for training the critic.")
    parser.add_argument("--critic_hidden_layers", type=int,
                        default=1, help="Number of hidden layers the critic has.")
    parser.add_argument("--critic_hidden_units", type=int,
                        default=300, help="Number of hidden units the critic has.")
    parser.add_argument("--adversarial_lambda", type=float, default=1.0,
                        help="Weight of adversarial loss. Decrease to reduce the adversarial loss term's influence.")
    parser.add_argument("--unaligned", action="store_true",
                        help="If set, input and desired output to the basemodel are the same.")
    # reproducibility
    parser.add_argument("--seed", type=int, default=1234, help="seed")

    # data
    parser.add_argument("--embedding_dim", type=int,
                        default=1024, help="sentence embedding dimension")
    parser.add_argument("--data_fraction", type=float,
                        default=1., help="How much of the data to use.")
    parser.add_argument("--print_outputs", action="store_true",
                        help="Print some of the outputs at validation time for inspection.")
    parser.add_argument("--max_prints", type=int, default=5,
                        help="How many examples to print during validation time.")
    parser.add_argument("--log_freq", type=int, default=100,
                        help="How often to print the logs.")
    parser.add_argument("--eval_self_bleu", action="store_true",
                        help="Whether to compute self-bleu scores on WikiLarge.")
    parser.add_argument("--invert_style", action="store_true",
                        help="Whether to invert the style transfer task (Yelp).")
    return parser


def get_params():
    parser = get_train_parser()
    params, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        raise ValueError("Got unknown parameters " + str(unknown))
    return params


def get_encoder(params, device, model_state_dict=None):
    if params.autoencoder == "FromFile":
        config = {"modeldir": params.modeldir, "use_lookup": True,
                  "device": device, "default_config": DEFAULT_CONFIG}
        encoder = AEEncoder(config)
    else:
        raise ValueError(f"Unknown autoencoder '{params.autoencoder}")
    return encoder


def get_decoder(params, device, model_state_dict=None):
    if params.autoencoder == "FromFile":
        config = {"modeldir": params.modeldir,
                  "device": device, "default_config": DEFAULT_CONFIG}
        decoder = AEDecoder(config)
    else:
        raise ValueError(f"Unknown autoencoder '{params.autoencoder}")
    return decoder


def get_emb2emb(params, encoder, train):
    if params.mapping == "mlp":
        return MLP(params.embedding_dim, params.n_layers, params.hidden_layer_size, dropout_p=params.dropout_p)
    if params.mapping == "identity":
        return nn.Sequential()
    if params.mapping == "offsetnet":
        return OffsetNet(params.embedding_dim, params.n_layers,
                         dropout_p=params.dropout_p,
                         offset_dropout_p=params.offset_dropout_p)
    if params.mapping == "resnet":
        return ResNet(params.embedding_dim, params.n_layers,
                      dropout_p=params.dropout_p,
                      offset_dropout_p=params.offset_dropout_p)
    if params.mapping == "meanoffsetvector":
        return MeanOffsetVectorMLP(params.embedding_dim, params.meanoffsetvector_factor, encoder, train["Sx"], train["Sy"])


def get_lossfn(params, encoder, data):
    if params.loss == "mse":
        return MSELoss()
    elif params.loss == "cosine":
        return CosineLoss()
    elif params.loss == "ce":
        return CrossEntropyLoss(ignore_index=0)  # ignore padding symbol
    elif params.loss == "fliploss":
        if params.baseloss == "cosine":
            baseloss = CosineLoss()
        elif params.baseloss == "mse":
            baseloss = MSELoss()
        else:
            raise ValueError("Unknown base loss {params.baseloss}.")

        bclf = train_binary_classifier(data['Sx'], data['Sy'], encoder, params)
        params.latent_binary_classifier = bclf
        return FlipLoss(baseloss, bclf,
                        lambda_clfloss=params.lambda_clfloss)


def get_mode(params):
    return params.mode


def _load_real_data(real_data_file):
    data = []
    with open(real_data_file, 'r') as f:
        for l in f:
            data.append(l.strip())

    return data


def configure_fgim(params, emb2emb):

    # configure FGIM
    if params.fgim_use_training_loss:

        def loss_f(x, Y_embeddings):
            x = x.view(-1, x.size(2))
            target_y = Y_embeddings.detach().clone()
            target_y = target_y.unsqueeze(0)
            target_y = target_y.repeat(
                len(params.fgim_weights), 1, 1).view(-1, target_y.size(2))
            l = emb2emb.compute_loss(x, target_y)
            if type(l) == tuple:
                return l[0]
            else:
                return l
    else:
        bin_clf_loss = make_binary_classification_loss(
            1, params.latent_binary_classifier if hasattr(
                params, 'latent_binary_classifier') else None)

        def loss_f(x, y): return bin_clf_loss(x)

    if params.fgim_no_stop_criterion:
        criterion_f = not_matched

    else:
        def criterion_f(x): return binary_classification_criterion(x,
                                                                   t=params.fgim_threshold,
                                                                   binary_classifier=params.latent_binary_classifier if hasattr(
                                                                       params, 'latent_binary_classifier') else None,
                                                                   target=1)

    return loss_f, criterion_f


def train(params):

    # set gpu device
    device = torch.device(params.device)
    print("Using device {}".format(str(device)))
    if "cuda" in params.device:
        print(torch.cuda.get_device_properties(device))

    # print parameters passed, and all parameters
    print('\ntogrep : {0}\n'.format(sys.argv[1:]))
    print(params)

    outputmodelname = params.outputmodelname + str(time.time())
    # save mapping model path for later use
    params.emb2emb_outputmodelname = outputmodelname
    """
    SEED
    """
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(params.seed)

    """
    DATA
    """
    (train, valid, test), eval_function = get_data(params)

    """
    Create the model.
    """
    # model
    encoder = get_encoder(params, device).to(device)
    decoder = get_decoder(params, device)
    emb2emb = get_emb2emb(params, encoder, train)
    loss_fn = get_lossfn(params, encoder, train)
    mode = get_mode(params)

    if params.unaligned:
        # set input and output of training the same
        train["Sy"] = train["Sx"]

    model = Emb2Emb(encoder, decoder, emb2emb, loss_fn, mode,
                    use_adversarial_term=params.adversarial_regularization,
                    adversarial_lambda=params.adversarial_lambda,
                    device=device,
                    critic_lr=params.critic_lr,
                    embedding_dim=params.embedding_dim,
                    critic_hidden_units=params.critic_hidden_units,
                    critic_hidden_layers=params.critic_hidden_layers,
                    real_data=params.real_data_path if params.real_data_path == "input" else _load_real_data(
                        params.real_data_path),
                    fast_gradient_iterative_modification=params.fast_gradient_iterative_modification,
                    fgim_decay=params.fgim_decay,
                    fgim_start_at_y=params.fgim_start_at_y
                    )

    if params.fast_gradient_iterative_modification:
        loss_f, criterion_f = configure_fgim(params, model)
        model.fgim_loss_f = loss_f
        model.fgim_criterion_f = criterion_f

    if params.load_emb2emb_path is not None:
        model.load_state_dict(torch.load(
            params.load_emb2emb_path)['model_state_dict'])

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    # cuda by default
    model.to(device)
    loss_fn.to(device)

    """
    TRAIN
    """
    val_acc_best = -1e10
    stop_training = False
    batch_counter = 0
    critic_losses = []
    params.time_for_epoch = 0

    def trainepoch(epoch):
        model.iterations = 0

        if (params.mapping == "identity" and params.mode != MODE_SEQ2SEQ) or params.lr == 0.:
            return 0.

        print('\nTRAINING : Epoch ' + str(epoch))
        model.train()
        all_costs = []
        logs = []
        nonlocal batch_counter, critic_losses

        last_time = time.time()
        # shuffle the data
        indices = list(range(len(train["Sx"])))
        random.shuffle(indices)

        Sx = [train['Sx'][i] for i in indices]
        Sy = [train['Sy'][i] for i in indices]

        start_epoch_time = time.time()

        for stidx in range(0, len(Sx), params.batch_size):
            batch_counter = batch_counter + 1

            # prepare batch
            Sx_batch = Sx[stidx:stidx + params.batch_size]
            Sy_batch = Sy[stidx:stidx + params.batch_size]

            k = len(Sx_batch)  # actual batch size

            with torch.autograd.set_detect_anomaly(True):
                # model forward
                if params.adversarial_regularization:

                    # forward pass
                    loss, task_loss, critic_loss, train_critic_loss = model(
                        Sx_batch, Sy_batch)
                    all_costs.append(
                        [loss.item(), task_loss.item(), critic_loss.item(), train_critic_loss.item()])
                    critic_losses.append(critic_loss.item())

                else:
                    loss = model(Sx_batch, Sy_batch)

                    # loss
                    all_costs.append(loss.item())

                # backward
                optimizer.zero_grad()
                loss.backward()

            # optimizer step
            optimizer.step()

            if len(all_costs) == params.log_freq:

                if not params.adversarial_regularization:
                    log_string = '{0} ; loss {1} ; sentence/s {2}'
                    log_string = log_string.format(
                        stidx, round(np.mean(all_costs), 5),
                        int(len(all_costs) * params.batch_size / (time.time() - last_time)))
                else:
                    mean_losses = np.reshape(
                        np.array(all_costs).mean(axis=0), (-1))
                    mean_losses = np.round(mean_losses, decimals=5)
                    log_string = '{0} ; loss {1} ; sentence/s {2} ; t-loss {3} ; c-loss {4} ; tc-loss {5}'
                    log_string = log_string.format(
                        stidx, mean_losses[0],
                        int(len(all_costs) * params.batch_size /
                            (time.time() - last_time)),
                        mean_losses[1], mean_losses[2], mean_losses[3])

                logs.append(log_string)
                print(logs[-1])
                # for p in model.mapping.parameters():
                #    print(p.grad)
                #    break
                last_time = time.time()
                all_costs = []

            if params.validation_frequency > 0 and (batch_counter % params.validation_frequency) == 0:
                evaluate(epoch, eval_type='valid', final_eval=False)
                model.train()

        params.time_for_epoch = time.time() - start_epoch_time
        return round(np.mean(all_costs), 5)

    def evaluate(epoch, eval_type='valid', final_eval=False):
        model.eval()

        if eval_function is not None:
            score = eval_function(
                model, mode="valid" if not final_eval else "test", params=params)
            print("Total Inference time", model.total_inference_time)
            print("Total Emb2Emb time", model.total_emb2emb_time)
            print("Total FGIM time", model.total_time_fgim)
            if type(score) == tuple:
                tmp_score = score
                score = tmp_score[0]
                self_bleu = tmp_score[1]
                b_acc = tmp_score[2]
            else:
                self_bleu = None
                b_acc = None

            if eval_type == 'valid':
                nonlocal val_acc_best

                if score > val_acc_best:
                    val_acc_best = max(val_acc_best, score)
                    checkpoint = {"model_state_dict": model.state_dict()}
                    torch.save(checkpoint, os.path.join(
                        params.outputdir, outputmodelname))
        else:

            if eval_type == 'valid':
                print('\nVALIDATION : Epoch {0}'.format(epoch))

            if eval_type == "valid":
                Sx = valid['Sx']
                Sy = valid['Sy']
            else:
                Sx = test['Sx']
                Sy = test['Sy']

            for stidx in range(0, len(Sx), params.batch_size):
                # prepare batch
                Sx_batch = Sx[stidx:stidx + params.batch_size]
                Sy_batch = Sy[stidx:stidx + params.batch_size]

                # model forward
                with torch.no_grad():
                    outputs = model(Sx_batch, Sy_batch)

                if params.print_outputs:
                    for i in range(len(Sx_batch[:5])):
                        input = Sx_batch[i]
                        gold_output = Sy_batch[i]
                        predicted_output = outputs[i]
                        pretty_print_prediction(
                            input, gold_output, predicted_output)

                    break
                else:
                    break
            score = 0

        eval_string = "Validation-Score in epoch {}/{} : {}; best : {}".format(
            epoch, batch_counter, score, val_acc_best)
        if b_acc is not None:
            eval_string = eval_string + " ; b-acc : {}".format(b_acc)
        if self_bleu is not None:
            eval_string = eval_string + " ; self-bleu : {}".format(self_bleu)
        print(eval_string)
        return score

    """
    Train model
    """
    epoch = 1

    while not stop_training and epoch <= params.n_epochs:
        train_loss = trainepoch(epoch)
        if params.adversarial_regularization:
            print('Epoch {0} ; loss {1} ; lambda {2}'.format(
                epoch, train_loss, model.adversarial_lambda))
        else:
            print('Epoch {0} ; loss {1}'.format(
                epoch, train_loss))

        if params.validate and params.validation_frequency < 0:
            evaluate(epoch, 'valid')
        epoch += 1

    # Run best model on test set.
    if params.validate:
        try:
            checkpoint = torch.load(os.path.join(
                params.outputdir, outputmodelname))
            model.load_state_dict(checkpoint["model_state_dict"])
        except:
            # no model saved so far
            pass

    results = {}
    if params.validate:
        final_val_score = evaluate(1e6, 'valid', False)
        results["dev"] = final_val_score
    final_test_score = evaluate(0, 'test', True)
    results["test"] = final_test_score
    return results


def pretty_print_prediction(input, gold_output, predicted_output):
    print("\n\n\n")
    print("Input: ", input)
    print("Output: ", predicted_output)
    print("Gold: ", gold_output)


if __name__ == "__main__":
    params = get_params()
    train(params)
