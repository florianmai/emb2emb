import csv
import os
from random import randint
from os.path import join
from sari.SARI import SARIsent
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


def read_all(path):
    with open(path, 'r') as f:
        all_examples = f.readlines()
        all_examples = [t.strip() for t in all_examples]
    return all_examples


def read_file(path, params):
    all_examples = read_all(path)

    num_examples = int(params.data_fraction * len(all_examples))
    all_examples = all_examples[:num_examples]
    return all_examples


def get_data(params):
    if "wiki" in params.dataset_path:
        params.run_id = randint(0, 999999999)

        # load the binary classifier
        if params.binary_classifier_path == "no_eval":
            params.binary_classifier = -1  # do not eval binary accuracy
        elif params.binary_classifier_path is not None:
            params.binary_tokenizer = AutoTokenizer.from_pretrained(
                params.binary_classifier_path)
            params.binary_classifier = AutoModelForSequenceClassification.from_pretrained(
                params.binary_classifier_path)
        else:
            params.binary_classifier = None
        params.current_epoch = 0
        return _get_data_pairs(params), evaluate_wiki
    elif "yelp" in params.dataset_path:
        params.run_id = randint(0, 999999999)

        # load the binary classifier
        if params.binary_classifier_path:
            params.binary_tokenizer = AutoTokenizer.from_pretrained(
                params.binary_classifier_path)
            params.binary_classifier = AutoModelForSequenceClassification.from_pretrained(
                params.binary_classifier_path)
        else:
            params.binary_classifier = None
        params.current_epoch = 0
        return _get_data_pairs(params), evaluate_yelp
    else:
        raise ValueError("Don't know dataset " + str(params.dataset_path))


def evaluate_yelp(model, mode="valid", params=None, predictions=None):

    # compute bleu with input
    if mode == "valid":
        data = "data/yelp/s1.dev" if not params.invert_style else "data/yelp/s2.dev"
    elif mode == "test":
        data = "data/yelp/s1.test" if not params.invert_style else "data/yelp/s2.test"

    inputs = data
    ref = data

    inputs = read_file(inputs, params)
    ref = read_file(ref, params)
    ref = [[r] for r in ref]
    self_bleu, predictions = evaluate_bleu(model, inputs, ref, params.batch_size,
                                           0 if not params.print_outputs else params.max_prints,
                                           return_predictions=True, predictions=predictions)
    b_acc = eval_binary_accuracy(model, predictions, mode, params)

    _save_to_csv(params, self_bleu=self_bleu, b_acc=b_acc)
    params.current_epoch = params.current_epoch + 1

    return self_bleu + b_acc, self_bleu, b_acc


def _save_to_csv(params, b_acc=None, sari=None, bleu=None, self_bleu=None):
    write_to_csv({"run_id": params.run_id,
                  "epoch": params.current_epoch,
                  "bleu": bleu,
                  "sari": sari,
                  "self-bleu": self_bleu,
                  "b-acc": b_acc},
                 params)


def write_to_csv(score, opt, escaped_keys=["binary_classifier", "binary_tokenizer", "latent_binary_classifier"]):
    """
    Writes the scores and configuration to csv file.
    """
    f = open(opt.output_file, 'a')
    if os.stat(opt.output_file).st_size == 0:
        for i, (key, _) in enumerate(opt.__dict__.items()):
            f.write(key + ";")
        for i, (key, _) in enumerate(score.items()):
            if i < len(score.items()) - 1:
                f.write(key + ";")
            else:
                f.write(key)
        f.write('\n')
        f.flush()
    f.close()

    f = open(opt.output_file, 'r')
    reader = csv.reader(f, delimiter=";")
    column_names = next(reader)
    f.close()

    def clean_str(s):
        return s.replace("\n", "")

    f = open(opt.output_file, 'a')
    for i, key in enumerate(column_names):
        if i < len(column_names) - 1:
            if key in opt.__dict__:
                if key in escaped_keys:
                    val_str = ""
                else:
                    val_str = str(opt.__dict__[key])
                    val_str = clean_str(val_str)
                f.write(val_str + ";")
            else:
                f.write(str(score[key]) + ";")
        else:
            if key in opt.__dict__:
                val_str = str(opt.__dict__[key])
                f.write(clean_str(val_str))
            else:
                f.write(str(score[key]))
    f.write('\n')
    f.flush()
    f.close()


def eval_binary_accuracy(model, predictions, mode="valid", params=None):
    target = 0 if params.invert_style else 1
    if params.binary_classifier is not None:

        if params.binary_classifier == -1:
            return 0.

        total_count = len(predictions)
        tokenizer = params.binary_tokenizer
        model = params.binary_classifier
        model.eval()
        correct = 0.
        for stidx in range(0, len(predictions), params.batch_size):
            # prepare batch
            predictions_batch = predictions[stidx:(stidx + params.batch_size)]

            predictions_batch = tokenizer.batch_encode_plus(
                predictions_batch, return_tensors="pt", pad_to_max_length=True)
            # returns logits, hidden_states
            predictions_batch = model(**predictions_batch)
            predictions_batch = predictions_batch[0]  # get logits

            predictions_batch = torch.softmax(predictions_batch, dim=1)
            predictions_batch = predictions_batch[:, target]
            b_acc = (predictions_batch > 0.5).sum()

            correct = correct + b_acc.item()

        return correct / float(total_count)
    else:

        model.eval()
        binary_classifier = model.loss_fn.classifier

        batch_size = params.batch_size
        target = 0  # we want to generate from the "fake distribution" labeled "0"
        correct = 0
        for stidx in range(0, len(predictions), batch_size):
            # prepare batch
            Sx_batch = predictions[stidx:stidx + batch_size]
            # model forward
            clf_predictions = model.compute_emb2emb(Sx_batch)[0]
            clf_predictions = torch.sigmoid(binary_classifier(clf_predictions))

            if target == 1:
                b_acc = (clf_predictions > 0.5).sum()
            elif target == 0:
                b_acc = (clf_predictions < 0.5).sum()
            correct = correct + b_acc.item()

        return correct / float(len(predictions))


def bleu_tokenize(s):
    return s.split()


def evaluate_bleu(model, input_sentences, reference_sentences, batch_size, max_prints, return_predictions=False, predictions=None):
    model.eval()

    if predictions is None:
        pred_outputs = _get_predictions(
            model, input_sentences, reference_sentences, batch_size, max_prints)
    else:
        pred_outputs = predictions

    # corpus_bleu(list_of_references, hypotheses) # list_of_refereces : list
    # of list of list of str, hypotheses list of list of str
    list_of_references = []
    for refs in reference_sentences:
        new_refs = []
        for r in refs:
            new_refs.append(bleu_tokenize(r))
        list_of_references.append(new_refs)

    pred_outputs_bleu = [bleu_tokenize(h) for h in pred_outputs]
    score = corpus_bleu(list_of_references, pred_outputs_bleu,
                        smoothing_function=SmoothingFunction().method1)
    if return_predictions:
        return score, pred_outputs
    else:
        return score


def _get_predictions(model, input_sentences, reference_sentences, batch_size, max_prints):
    model.eval()

    pred_outputs = []
    i = 1
    for i, stidx in enumerate(range(0, len(input_sentences), batch_size)):
        if i % 10 == 0:
            print("Eval progress:", float(stidx) / len(input_sentences))

        # prepare batch
        Sx_batch = input_sentences[stidx:stidx + batch_size]
        Sy_batch = reference_sentences[stidx:stidx + batch_size][0]
        # model forward
        with torch.no_grad():
            pred_outputs.extend(model(Sx_batch, Sy_batch))

    for i in range(min(len(input_sentences), max_prints)):
        pretty_print_prediction(
            input_sentences[i], reference_sentences[i][0], pred_outputs[i])

    return pred_outputs


def evaluate_wiki(model, mode="valid", params=None):

    sari, predictions = evaluate_sari(model, mode, params)
    b_acc = eval_binary_accuracy(model, predictions, mode, params)

    reference_sentences, norm_sentences, _ = _load_wikilarge_references(mode)
    bleu = evaluate_bleu(model, norm_sentences, reference_sentences, params.batch_size,
                         max_prints=0, return_predictions=False, predictions=predictions)
    if params.eval_self_bleu:
        self_bleu = evaluate_bleu(model, norm_sentences, [
                                  [n] for n in norm_sentences], params.batch_size, max_prints=0, return_predictions=False, predictions=predictions)
    else:
        self_bleu = -1.

    _save_to_csv(params, b_acc=b_acc, sari=sari,
                 bleu=bleu, self_bleu=self_bleu)
    params.current_epoch = params.current_epoch + 1

    return sari, sari, b_acc


def _load_wikilarge_references(mode):
    if mode == "valid":
        base_path = "./data/simplification/valid/"
    elif mode == "test":
        base_path = "./data/simplification/test/"

    norm_sentences = read_all(join(base_path, "norm"))
    simp_sentences = read_all(join(base_path, "simp"))

    reference_sentences_sep = [
        read_all(join(base_path, "turk" + str(i))) for i in range(8)]
    reference_sentences = []
    for i in range(len(reference_sentences_sep[0])):
        reference_sentences.append(
            [reference_sentences_sep[j][i] for j in range(8)])

    return reference_sentences, norm_sentences, simp_sentences


def evaluate_sari(model, mode="valid", params=None):
    batch_size = params.batch_size

    model.eval()

    reference_sentences, norm_sentences, simp_sentences = _load_wikilarge_references(
        mode)

    pred_simple_sentences = []
    for stidx in range(0, len(norm_sentences), batch_size):
        # prepare batch
        Sx_batch = norm_sentences[stidx:stidx + batch_size]
        Sy_batch = simp_sentences[stidx:stidx + batch_size]
        # model forward
        with torch.no_grad():
            pred_simple_sentences.extend(model(Sx_batch, Sy_batch))

    copy_baseline = _calc_sari(
        norm_sentences, norm_sentences, reference_sentences, params)
    obtained_scores = _calc_sari(
        norm_sentences, pred_simple_sentences, reference_sentences, params)
    print("Text Simplification Copy-Baseline:", copy_baseline)
    return obtained_scores, pred_simple_sentences


def _calc_sari(norm_sentences, pred_simple_sentences, reference_sentences, params):
    sari_scores = []
    for i, (n, s, rs) in enumerate(zip(norm_sentences, pred_simple_sentences, reference_sentences)):

        sari_scores.append(SARIsent(n, s, rs))
        if params.print_outputs and i < params.max_prints:
            pretty_print_prediction(n, rs[0], s)

    return np.array(sari_scores).mean()


def _get_data_pairs(params):
    """
    The dataset is assumed to be given as a directory containing
    the files 's1' (input sequence) and 's2' (output sequence) for each of the
    data splits, i.e. 's1.train', 's1.dev', 's1.test', and 's2.train', 's2.dev',
    's2.test'.
    Each file contains one text per line.
    """
    dataset_path = params.dataset_path

    endings = ["train", "dev", "test"]
    data_dict = {e: {} for e in endings}
    for ending in endings:
        s1 = read_file(join(dataset_path, "s1." + ending), params)
        s2 = read_file(join(dataset_path, "s2." + ending), params)
        data_dict[ending]["Sx"] = s1 if not params.invert_style else s2
        data_dict[ending]["Sy"] = s2 if not params.invert_style else s1
    return data_dict["train"], data_dict["dev"], data_dict["test"]


def pretty_print_prediction(input_text, gold_output, predicted_output):
    print("\n\n\n")
    print("Input: ", input_text)
    print("Output: ", predicted_output)
    print("Gold: ", gold_output)
