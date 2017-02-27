# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils import UNK_ID
import copy

def initial_state(net, sess):
    # Return freshly initialized model states.
    return sess.run(net.cell.zero_state(1, tf.float32))

def forward_text(net, sess, states, vocab, prime_text=None):
    if prime_text is not None:
        for char in prime_text:
            if len(states) == 2:
                # Automatically forward the primary net.
                _, states[0] = net.forward_model(sess, states[0], vocab.get(char, UNK_ID))
                # If the token is newline, reset the mask net state; else, forward it.
                if vocab.get(char, UNK_ID) == '\n':
                    states[1] = initial_state(net, sess)
                else:
                    _, states[1] = net.forward_model(sess, states[1], vocab.get(char, UNK_ID))
            else:
                _, states = net.forward_model(sess, states, vocab.get(char, UNK_ID))
    return states

def beam_sample(net, sess, chars, vocab, max_length=200, prime='The ',
                beam_width = 2, relevance=3.0, temperature=1.0):
    prime = prime.decode('utf-8')
    states = [initial_state(net, sess), initial_state(net, sess)]
    states = forward_text(net, sess, states, vocab, prime[:-1])
    computer_response_generator = beam_search_generator(sess=sess, net=net,
            initial_state=copy.deepcopy(states), initial_sample=vocab[prime[-1]],
            early_term_token=vocab['\n'], beam_width=beam_width, forward_model_fn=forward_with_mask,
            forward_args=(relevance, vocab['\n']), temperature=temperature)
    ret = prime[:-1]
    for i, char_token in enumerate(computer_response_generator):
        ret = ret + chars[char_token]
        states = forward_text(net, sess, states, vocab, chars[char_token])
        if i >= max_length: break
    return ret

def forward_with_mask(sess, net, states, input_sample, forward_args):
    if len(states) != 2:
        # No relevance masking.
        prob, states = net.forward_model(sess, states, input_sample)
        return prob / sum(prob), states
    # states should be a 2-length list: [primary net state, mask net state].
    # forward_args should be a 2-length list/tuple: [relevance, mask_reset_token]
    relevance, mask_reset_token = forward_args
    if input_sample == mask_reset_token:
        # Reset the mask probs when reaching mask_reset_token (newline).
        states[1] = initial_state(net, sess)
    primary_prob, states[0] = net.forward_model(sess, states[0], input_sample)
    primary_prob /= sum(primary_prob)
    mask_prob, states[1] = net.forward_model(sess, states[1], input_sample)
    mask_prob /= sum(mask_prob)
    combined_prob = np.exp(np.log(primary_prob) - relevance * np.log(mask_prob))
    # Normalize probabilities so they sum to 1.
    return combined_prob / sum(combined_prob), states

def scale_prediction(prediction, temperature):
    if (temperature == 1.0): return prediction # Temperature 1.0 makes no change
    np.seterr(divide='ignore')
    scaled_prediction = np.log(prediction) / temperature
    scaled_prediction = scaled_prediction - np.logaddexp.reduce(scaled_prediction)
    scaled_prediction = np.exp(scaled_prediction)
    np.seterr(divide='warn')
    return scaled_prediction

def beam_search_generator(sess, net, initial_state, initial_sample,
    early_term_token, beam_width, forward_model_fn, forward_args, temperature):
    '''Run beam search! Yield consensus tokens sequentially, as a generator;
    return when reaching early_term_token (newline).
    Args:
        sess: tensorflow session reference
        net: tensorflow net graph (must be compatible with the forward_net function)
        initial_state: initial hidden state of the net
        initial_sample: single token (excluding any seed/priming material)
            to start the generation
        early_term_token: stop when the beam reaches consensus on this token
            (but do not return this token).
        beam_width: how many beams to track
        forward_model_fn: function to forward the model, must be of the form:
            probability_output, beam_state =
                    forward_model_fn(sess, net, beam_state, beam_sample, forward_args)
            (Note: probability_output has to be a valid probability distribution!)
        temperature: how conservatively to sample tokens from each distribution
            (1.0 = neutral, lower means more conservative)
        tot_steps: how many tokens to generate before stopping,
            unless already stopped via early_term_token.
    Returns: a generator to yield a sequence of beam-sampled tokens.'''
    # Store state, outputs and probabilities for up to args.beam_width beams.
    # Initialize with just the one starting entry; it will branch to fill the beam
    # in the first step.
    beam_states = [initial_state] # Stores the best activation states
    beam_outputs = [[initial_sample]] # Stores the best generated output sequences so far.
    beam_probs = [1.] # Stores the cumulative normalized probabilities of the beams so far.

    while True:
        # Keep a running list of the best beam branches for next step.
        # Don't actually copy any big data structures yet, just keep references
        # to existing beam state entries, and then clone them as necessary
        # at the end of the generation step.
        new_beam_indices = []
        new_beam_probs = []
        new_beam_samples = []

        # Iterate through the beam entries.
        for beam_index, beam_state in enumerate(beam_states):
            beam_prob = beam_probs[beam_index]
            beam_sample = beam_outputs[beam_index][-1]

            # Forward the model.
            prediction, beam_states[beam_index] = forward_model_fn(
                    sess, net, beam_state, beam_sample, forward_args)
            prediction = scale_prediction(prediction, temperature)

            # Sample best_tokens from the probability distribution.
            # Sample from the scaled probability distribution beam_width choices
            # (but not more than the number of positive probabilities in scaled_prediction).
            count = min(beam_width, sum(1 if p > 0. else 0 for p in prediction))
            best_tokens = np.random.choice(len(prediction), size=count,
                                            replace=False, p=prediction)
            for token in best_tokens:
                prob = prediction[token] * beam_prob
                if len(new_beam_indices) < beam_width:
                    # If we don't have enough new_beam_indices, we automatically qualify.
                    new_beam_indices.append(beam_index)
                    new_beam_probs.append(prob)
                    new_beam_samples.append(token)
                else:
                    # Sample a low-probability beam to possibly replace.
                    np_new_beam_probs = np.array(new_beam_probs)
                    inverse_probs = -np_new_beam_probs + max(np_new_beam_probs) + min(np_new_beam_probs)
                    inverse_probs = inverse_probs / sum(inverse_probs)
                    sampled_beam_index = np.random.choice(beam_width, p=inverse_probs)
                    if new_beam_probs[sampled_beam_index] <= prob:
                        # Replace it.
                        new_beam_indices[sampled_beam_index] = beam_index
                        new_beam_probs[sampled_beam_index] = prob
                        new_beam_samples[sampled_beam_index] = token
        # Replace the old states with the new states, first by referencing and then by copying.
        already_referenced = [False] * beam_width
        new_beam_states = []
        new_beam_outputs = []
        for i, new_index in enumerate(new_beam_indices):
            if already_referenced[new_index]:
                new_beam = copy.deepcopy(beam_states[new_index])
            else:
                new_beam = beam_states[new_index]
                already_referenced[new_index] = True
            new_beam_states.append(new_beam)
            new_beam_outputs.append(beam_outputs[new_index] + [new_beam_samples[i]])
        # Normalize the beam probabilities so they don't drop to zero
        beam_probs = new_beam_probs / sum(new_beam_probs)
        beam_states = new_beam_states
        beam_outputs = new_beam_outputs
        # Prune the agreed portions of the outputs
        # and yield the tokens on which the beam has reached consensus.
        l, early_term = consensus_length(beam_outputs, early_term_token)
        if l > 0:
            for token in beam_outputs[0][:l]: yield token
            beam_outputs = [output[l:] for output in beam_outputs]
        if early_term: return

def consensus_length(beam_outputs, early_term_token):
    for l in range(len(beam_outputs[0])):
        if l > 0 and beam_outputs[0][l-1] == early_term_token:
            return l-1, True
        for b in beam_outputs[1:]:
            if beam_outputs[0][l] != b[l]: return l, False
    return l, False
