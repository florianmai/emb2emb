import os
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import random

random.seed(0)
from random import shuffle
import json
import copy
import torch
import time
import argparse
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from autoencoders.autoencoder import AutoEncoder
from autoencoders.rnn_decoder import RNNDecoder
from autoencoders.rnn_encoder import RNNEncoder
from emb2emb.utils import Namespace
import numpy as np
from tqdm import tqdm
from autoencoders.data_loaders import HDF5Dataset, get_tokenizer

DEFAULT_CONFIG = "autoencoders/config/default.json"
LOG_DIR_NAME = "logs/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="config/default.json",
                        help="The config file specifying all params.")
    params = parser.parse_args()
    with open(DEFAULT_CONFIG) as f:
        config = json.load(f)
    with open(params.config) as f:
        config.update(json.load(f))
    n = Namespace()
    n.__dict__.update(config)
    return n


def train_batch(model, optimizer, X, X_lens, lambda_r=1, lambda_kl=1, lambda_a=1):
    # Train autoencoder
    model.train()
    output = model(X, X_lens)

    # Won't be both adversarial and variational
    if model.variational:
        output, mu, z, embeddings = output
        loss, r_loss, kl_loss = model.loss_variational(
            output, embeddings, X, mu, z, lambda_r, lambda_kl)
    elif model.adversarial:
        output, fake_z_g, fake_z_d, true_z, embeddings = output
        loss, r_loss, d_loss, g_loss = model.loss_adversarial(
            output, embeddings, X, fake_z_g, fake_z_d, true_z, lambda_a)

        # update the discriminator independently
        model.optimD.zero_grad()
        d_loss.backward()
        model.optimD.step()
    else:
        predictions, embeddings = output
        loss = model.loss(predictions, embeddings, X)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if model.variational:
        return loss, r_loss, kl_loss
    elif model.adversarial:
        return loss, r_loss, d_loss, g_loss
    else:
        return loss


def test_batch(model, X, X_lens, lambda_r=1, lambda_kl=1, lambda_a=1):
    with torch.no_grad():
        output = model(X, X_lens)
        if model.variational:
            output, mu, z, embeddings = output
            loss, r_loss, kl_loss = model.loss_variational(
                output, embeddings, X, mu, z, lambda_r, lambda_kl)
            return loss, r_loss, kl_loss
        elif model.adversarial:
            output, fake_z_g, fake_z_d, true_z, embeddings = output
            loss, r_loss, d_loss, g_loss = model.loss_adversarial(
                output, embeddings, X, fake_z_g, fake_z_d, true_z, lambda_a)
            return loss, r_loss, d_loss, g_loss
        else:
            p, e = output
            loss = model.loss(p, e, X)
            return loss


def prepare_batch(indexed, lengths, device):
    X = pad_sequence([index_list.to(device)
                      for index_list in indexed], batch_first=True, padding_value=0)
    X = X[:, :lengths.max()]
    lengths, idx = torch.sort(lengths.to(device), descending=True)

    return X[idx], lengths


def evaluate(data, device, batch_size, lambda_r=1, lambda_kl=1, lambda_a=1):
    valid_losses = []
    valid_r_losses = []
    valid_d_losses = []
    valid_g_losses = []
    valid_kl_losses = []
    for data_b, lens_b in data:
        X_valid, X_valid_lens = prepare_batch(data_b, lens_b, device)
        valid_loss = test_batch(
            model, X_valid, X_valid_lens, lambda_r, lambda_kl, lambda_a)
        if model.variational:
            loss, r_loss, kl_loss = valid_loss
            valid_losses.append(loss.cpu().detach().numpy().item())
            valid_r_losses.append(r_loss.cpu().detach().numpy().item())
            valid_kl_losses.append(kl_loss.cpu().detach().numpy().item())
        elif model.adversarial:
            loss, r_loss, d_loss, g_loss = valid_loss
            valid_losses.append(loss.cpu().detach().numpy().item())
            valid_r_losses.append(r_loss.cpu().detach().numpy().item())
            valid_d_losses.append(d_loss.cpu().detach().numpy().item())
            valid_g_losses.append(g_loss.cpu().detach().numpy().item())
        else:
            valid_losses.append(valid_loss.cpu().detach().numpy().item())
    if model.variational:
        return np.array(valid_losses).mean(axis=0), np.array(valid_r_losses).mean(axis=0), np.array(valid_kl_losses).mean(axis=0)
    elif model.adversarial:
        return np.array(valid_losses).mean(axis=0), np.array(valid_r_losses).mean(axis=0), np.array(valid_d_losses).mean(axis=0), np.array(valid_g_losses).mean(axis=0)
    else:
        return np.array(valid_losses).mean(axis=0)


def eval(model, X, X_lens, noise, device):
    encoded = model.encode(X, X_lens)
    if noise != 0.0:
        encoded += torch.randn_like(encoded, device=device) * noise
    return (model.beam_decode(encoded), model.greedy_decode(encoded))


def evaluate_sentence(model, data, device, tokenizer):
    with torch.no_grad():
        for data_b, lens_b in data:
            X, X_lens = prepare_batch(data_b[:1], lens_b[:1], device)
            encoded = model.encode(X, X_lens)
            greedy, beam = model.decode(
                encoded), model.decode(encoded, beam_width=10)
            return {"original": tokenizer.decode(X[0].tolist()),
                    "greedy": tokenizer.decode(greedy[0]),
                    "beam": tokenizer.decode(beam[0])}


def get_model_info(config):
    model_info = copy.deepcopy(config.__dict__)
    for key in list(model_info):
        if isinstance(model_info[key], dict):
            if key == config.encoder:
                e_info = model_info[key]
                for kk in e_info:
                    model_info["e_" + kk] = e_info[kk]
            elif key == config.decoder:
                d_info = model_info[key]
                for kk in d_info:
                    model_info["d_" + kk] = d_info[kk]
            del model_info[key]
    return model_info


if __name__ == "__main__":
    config = parse_args()

    original_config = copy.deepcopy(config)

    print(json.dumps(config.__dict__, indent=4))

    model_info = get_model_info(config)

    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu")
    print(device)

    # set config for encoders
    tokenizer = get_tokenizer(config.tokenizer, config.tokenizer_location)
    config.__dict__["vocab_size"] = tokenizer.get_vocab_size()
    config.__dict__["sos_idx"] = tokenizer.token_to_id("<SOS>")
    config.__dict__["eos_idx"] = tokenizer.token_to_id("<EOS>")
    config.__dict__["unk_idx"] = tokenizer.token_to_id("<unk>")

    config.__dict__["device"] = device

    encoder_config, decoder_config = copy.deepcopy(
        config), copy.deepcopy(config)
    encoder_config.__dict__.update(config.__dict__[config.encoder])
    encoder_config.__dict__["tokenizer"] = tokenizer
    decoder_config.__dict__.update(config.__dict__[config.decoder])

    if config.encoder == "RNNEncoder":
        encoder = RNNEncoder(encoder_config)
    else:
        raise ValueError(
            f"Training configuration contains unknown encoder {config.encoder}.")

    if config.decoder == "RNNDecoder":
        decoder = RNNDecoder(decoder_config)
    else:
        raise ValueError(
            f"Training configuration contains unknown decoder {config.decoder}.")

    model = AutoEncoder(encoder, decoder, tokenizer, config)

    model_path = os.path.join(config.savedir, config.model_file)
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.train()

    if config.adversarial:
        model_parameters = []
        for name, param in model.named_parameters():
            if name.startswith("encoder") or name.startswith("decoder"):
                model_parameters.append(param)
            elif name.startswith("discriminator"):
                pass
            else:
                raise AssertionError(
                    "Found a model parameter " + name + " that we do not know how to handle.")
    else:
        model_parameters = model.parameters()
    optimizer = optim.Adam(model_parameters, lr=config.lr)
    if os.path.isfile(model_path):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(model)

    logdir = os.path.join(config.savedir, LOG_DIR_NAME)
    log_num = 0
    while os.path.isdir(os.path.join(logdir, str(log_num))):
        log_num += 1
    logdir = os.path.join(logdir, str(log_num))

    dataset = HDF5Dataset(config.dataset_path, False, False,
                          data_cache_size=3, transform=None)
    indices = list(range(len(dataset)))
    shuffle(indices)
    num_val_samples = int(len(indices) * config.valsize)
    train_indices = indices[:-num_val_samples]
    val_indices = indices[-num_val_samples:]

    # downsample if appropriate
    train_indices = train_indices[:int(
        config.data_fraction * len(train_indices))]
    val_indices = val_indices[:int(config.data_fraction * len(val_indices))]

    def collate_batches(batch):

        # 'batch' is a list of pairs (X, X_len) which are of size
        # [batch_size, max_len] and [batch_size], respectively. The default
        # collate_batches would create a new dimension, but we want to stack
        # alongside the batch_dimension.

        Xs, X_lens = zip(*batch)
        X = torch.cat(Xs, dim=0)
        X_len = torch.cat(X_lens, dim=0)

        return X, X_len

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=SubsetRandomSampler(
        train_indices), num_workers=config.workers, collate_fn=collate_batches)
    valloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=SubsetRandomSampler(
        val_indices), num_workers=config.workers, collate_fn=collate_batches)

    i = 0
    epoch = 0
    time_s = time.time()
    min_val_loss = float('inf')
    stop_training = False

    epoch_batches = len(trainloader)
    print(f"Epoch batches: {epoch_batches}")
    with SummaryWriter(log_dir=logdir) as sw:
        sw.add_hparams(model_info, {})

        if os.path.isfile(model_path):
            print("Running initial validation step.")
            val_loss = evaluate(
                valloader, device, 1, config.lambda_r, config.lambda_kl, config.lambda_a)
            if config.variational:
                val_loss, r_loss, kl_loss = val_loss
                sw.add_scalar(
                    "Variational Loss/Validation/Reconstruction", r_loss, i)
                sw.add_scalar("Variational Loss/Validation/KL", kl_loss, i)
            if config.adversarial:
                val_loss, r_loss, d_loss, g_loss = val_loss
                sw.add_scalar(
                    "Adversarial Loss/Validation/Reconstruction", r_loss, i)
                sw.add_scalar(
                    "Adversarial Loss/Validation/Discriminator", d_loss, i)
                sw.add_scalar(
                    "Adversarial Loss/Validation/Generator", g_loss, i)
            sw.add_scalar("Loss/Validation", val_loss, i)

            es = evaluate_sentence(model, valloader, device, tokenizer)
            sw.add_text("Validation/Original", es["original"], i)
            sw.add_text("Validation/Greedy", es["greedy"], i)
            sw.add_text("Validation/Beam", es["beam"], i)

            es = evaluate_sentence(model, trainloader, device, tokenizer)
            sw.add_text("Train/Original", es["original"], i)
            sw.add_text("Train/Greedy", es["greedy"], i)
            sw.add_text("Train/Beam", es["beam"], i)
            min_val_loss = val_loss

        print("Starting training")
        while not stop_training:
            pbar = tqdm(trainloader, desc=f"[E{epoch}, B{i}]")
            for s_batch, l_batch in pbar:
                i += 1

                # Train on batch
                lambda_kl = 0 if epoch < config.kl_delay else config.lambda_kl if epoch > config.kl_delay else config.lambda_kl * \
                    ((i - epoch_batches * epoch) / epoch_batches)
                X, X_lens = prepare_batch(s_batch, l_batch, device)
                actual_batch_size = X_lens.size(0)
                loss = train_batch(model, optimizer, X, X_lens, lambda_r=config.lambda_r,
                                   lambda_kl=lambda_kl, lambda_a=config.lambda_a)

                if config.variational:
                    loss, r_loss, kl_loss = loss
                if config.adversarial:
                    loss, r_loss, d_loss, g_loss = loss

                msg = ""
                if i % (config.print_frequency) == 0:
                    sw.add_scalar("Loss/Train", loss.cpu().item(), i)

                    msg = f"[E{epoch}, B{i}] tr={loss:0.2f}"
                    msg += f", val={min_val_loss:0.2f}"
                    if config.variational:
                        msg += f", r={r_loss:0.2f}, kl={kl_loss:0.2f}"
                        sw.add_scalar(
                            "Variational Loss/Train/Reconstruction", r_loss.cpu().item(), i)
                        sw.add_scalar("Variational Loss/Train/KL",
                                      kl_loss.cpu().item(), i)
                        sw.add_scalar(
                            "Variational Weight/Lambda KL", lambda_kl, i)
                        sw.add_scalar(
                            "Variational Weight/Lambda R", config.lambda_r, i)
                    if config.adversarial:
                        msg += f", r={r_loss:0.2f}, d={d_loss:0.2f}, g={g_loss:0.2f}"
                        sw.add_scalar(
                            "Adversarial Loss/Train/Reconstruction", r_loss.cpu().item(), i)
                        sw.add_scalar(
                            "Adversarial Loss/Train/Discriminator", d_loss.cpu().item(), i)
                        sw.add_scalar(
                            "Adversarial Loss/Train/Generator", g_loss.cpu().item(), i)

                    speed = ((config.print_frequency *
                              actual_batch_size) // (time.time() - time_s))
                    time_s = time.time()
                    sw.add_scalar("Speed/Speed", speed, i)
                    # print(msg)
                    pbar.set_description(msg)
                    sw.flush()

                # Validation
                if (i % config.validation_frequency) == 0:
                    val_loss = evaluate(
                        valloader, device, 1, config.lambda_r, config.lambda_kl, config.lambda_a)
                    if config.variational:
                        val_loss, r_loss, kl_loss = val_loss
                        sw.add_scalar(
                            "Variational Loss/Validation/Reconstruction", r_loss, i)
                        sw.add_scalar(
                            "Variational Loss/Validation/KL", kl_loss, i)
                    if config.adversarial:
                        val_loss, r_loss, d_loss, g_loss = val_loss
                        msg += f", r={r_loss:0.2f}, d={d_loss:0.2f}, g={g_loss:0.2f}"
                        sw.add_scalar(
                            "Adversarial Loss/Validation/Reconstruction", r_loss, i)
                        sw.add_scalar(
                            "Adversarial Loss/Validation/Discriminator", d_loss, i)
                        sw.add_scalar(
                            "Adversarial Loss/Validation/Generator", g_loss, i)

                    sw.add_scalar("Loss/Validation", val_loss, i)

                    es = evaluate_sentence(model, valloader, device, tokenizer)
                    sw.add_text("Validation/Original", es["original"], i)
                    sw.add_text("Validation/Greedy", es["greedy"], i)
                    sw.add_text("Validation/Beam", es["beam"], i)

                    es = evaluate_sentence(
                        model, trainloader, device, tokenizer)
                    sw.add_text("Train/Original", es["original"], i)
                    sw.add_text("Train/Greedy", es["greedy"], i)
                    sw.add_text("Train/Beam", es["beam"], i)

                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        os.makedirs(config.savedir, exist_ok=True)
                        checkpoint = {"model_state_dict": model.state_dict(),
                                      "optimizer_state_dict": optimizer.state_dict()}
                        torch.save(checkpoint, model_path)
                        with open(os.path.join(config.savedir, 'config.json'), 'w') as f:
                            json.dump(original_config.__dict__, f)

                if i == config.max_steps:
                    stop_training = True
                    break
            epoch += 1
