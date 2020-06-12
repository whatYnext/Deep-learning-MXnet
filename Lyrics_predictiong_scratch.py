# -*- coding: utf-8 -*-
"""
Using RNN to train on Jay's lyrics and predict lyrics under given
prefix

@author: mayao
"""
import d2lzh as d2l
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = d2l.load_data_jay_lyrics()

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()
print('will use', ctx)

#Change (batchsize, num_step) to num_step (batchsize, vacasize)
def to_onehot(X, size): 
    return [nd.one_hot(x, size) for x in X.T]

#An example input (batchsize, num_step)
#X = nd.arange(10).reshape((2, 5))

def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)
    # hidden parameters
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # output parameters 
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    # attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params

#Hidden state: (batchsize, num_hiddens)
#num_hidden is super parameter
def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )

def rnn(inputs, state, params):
    # inputs and outputs are num_step (batchsize, vacasize)
    # Use tanh as activation function
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)

#'''To print shape of an example'''
# state = init_rnn_state(X.shape[0], num_hiddens, ctx)
# inputs = to_onehot(X.as_in_context(ctx), vocab_size)
# params = get_params()
# outputs, state_new = rnn(inputs, state, params)
# print(len(outputs)), print(outputs[0].shape), print(state_new[0].shape)

# Given prefix to predict following words
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # Take last output as input
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        # Compute output and hidden state
        (Y, state) = rnn(X, state, params)
        # Next input is word in prefix or best predicted word
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])


# Because of gradient explosure or vanish, we define clip gradient.
def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# Detailed implementation of train and predict for Jay chou's lyrics
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # If using adjacent sampling, initiate hidden
                                # state at first
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter:  # if using random sampling, iniate at
                                # every batch size
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:  # Or extract state using detach()
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                # outputs is num_steps (batch_size, vocab_size)
                (outputs, state) = rnn(inputs, state, params)
                # Concat as (num_steps * batch_size, vocab_size)
                outputs = nd.concat(*outputs, dim=0)
                # Y from (batch_size, num_steps) to 
                # batch * num_steps for loss function's computation
                y = Y.T.reshape((-1,))
                # Use cross entropy to be loss function 
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)  # clipp gradient
            d2l.sgd(params, lr, 1)  # no need to be mean
            l_sum += l.asscalar() * y.size
            n += y.size
        # Print results using perplexity and also predict the results
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))
# Settings 
params = get_params()
num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 25, 20, 1e2, 1e-2
pred_period, pred_len, prefixes = 20, 20, ['为你爱过吗', '对这个世界']

# Start training and predicting
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)

