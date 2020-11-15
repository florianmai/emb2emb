import torch
from torch.nn import BCEWithLogitsLoss
import numpy as np


def not_matched(x):
    """
    This function can be used as `criterion_function` argument to
    function:`emb2emb.fgim.fast_gradient_iterative_modification`. It will always return zero (False),
    resulting in FGIM running for the specified number of iterations.
    """
    return torch.zeros((x.size(0), x.size(1)), device=x.device)


def binary_classification_criterion(x, t=0.001, binary_classifier=None, target=1):
    """
    This function can be used as a `criterion_function` argument to 
    function:`emb2emb.fgim.fast_gradient_iterative_modification`. It returns True
    if the input `x` is classified as `target` with probability 1 - `t` by the 
    `binary_classifier`.
    """
    logits = binary_classifier(x)
    preds = torch.sigmoid(logits).squeeze()

    correct_classification = torch.abs(target - preds) <= t
    return correct_classification


def _first_nonzero(x, axis=0):
    nonz = (x > 0)
    return ((nonz.cumsum(axis) == 1) & nonz).max(axis)


def make_binary_classification_loss(target, binary_classifier):
    """
    This function can be used to create a `loss_function` argument to 
    function:`emb2emb.fgim.fast_gradient_iterative_modification`. The loss function
    computes Binary Cross Entropy of the input being classified as `target` by 
    the `binary_classifier`.
    """
    def f(x):
        return _binary_classification_loss(x, target, binary_classifier)

    return f


def _binary_classification_loss(inputs, target, binary_classifier):

    # make sure gradients from previous iterations are not taken into account
    binary_classifier.zero_grad()

    # compute predictions on binary classifier
    logits = binary_classifier(inputs)
    preds = torch.sigmoid(logits).squeeze()

    # compute loss based on how far we are from the desired label
    desired_label = torch.ones_like(
        preds, device=inputs.device) if target == 1 else torch.zeros_like(preds, device=inputs.device)
    loss_f = BCEWithLogitsLoss()
    loss = loss_f(preds, desired_label)
    return loss


def fast_gradient_iterative_modification(inputs,
                                         loss_function,
                                         criterion_function,
                                         weights=[1.0, 2.0, 3.0,
                                                  4.0, 5.0, 6.0],
                                         decay_factor=1.0,
                                         max_steps=30):
    """
    Fast Gradient Iterative Modification (FGIM) is a technique for modifying a given
    embedding such that it satisfies some (differentiable) objective (`loss_function`) to a
    sufficient degree (threshold `criterion_function`), but at the same time doesn't deviate too
    far from the input embedding.

    FGIM works by iteratively following the gradient of `loss_function` with respect to the `inputs`.
    To avoid making too large gradient steps (and thus deviating too far from the input),
    the gradient steps are taken with different step sizes (`weights`) in parallel.
    The result of the smallest weight that first satisfies some criterion (`criterion_function`) is returned.
    The complete algorithm is introduced and described in the paper 
    `Controllable Unsupervised Text Attribute Transfer via Editing Entangled 
    Latent Representation <https://arxiv.org/pdf/1905.12926.pdf>`_.

    @param inputs: Batch of inputs that serve as the starting point of the FGIM search.
    @type inputs: Tensor
    @param loss_function: A function of the form f(x) -> R of which to compute 
        the gradient with respect to x. Output must be a tensor to call backward() on.
    @param criterion_function: A function of the form c(x) -> o, where x is a tensor
        of shape [batch_size, num_weights, embedding_size] and o is a tensor of the same shape
        that contains truth values (1/0) signaling whether the criterion is met.
    @param weights: Step sizes to apply at each iteration.
    @type weights: list of float
    @param decay_factor: The factor by which to multiply the weights after each iteration.
    @type decay_factor: float 
    @param max_steps: Maximum number of iterations before stopping FGIM.
    @type max_steps: int
    """
    with torch.enable_grad():

        weights = torch.tensor(np.array(weights), device=inputs.device).float()

        x = inputs.detach().clone()
        x = x.unsqueeze(0)
        x = x.repeat(len(weights), 1, 1)

        weights = weights.unsqueeze(1).unsqueeze(1)
        cnt = 0

        while True:
            cnt = cnt + 1
            x = x.detach().clone()
            x.requires_grad = True

            correct_classification = criterion_function(x)
            if (correct_classification.sum(dim=0) >= 1).all() or cnt == max_steps:
                _, first_success = _first_nonzero(correct_classification,
                                                  axis=0)
                break

            # compute the loss from which to obtain gradients
            loss = loss_function(x)

            weights = weights * decay_factor * \
                (1 - correct_classification.float().unsqueeze(2))

            loss.sum().backward()
            x = x - weights * x.grad

        # only output the result of the smallest weight that achieved the goal
        x = x.gather(0, first_success.view(1, -1, 1).expand_as(x))[0, :, :]

        return x
