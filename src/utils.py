#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Training and Evaluation utility functions
             
@author    : Reddy
"""

import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.nist_score import sentence_nist

MAX_LEN = 60 

def train(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio):
    """Train the model"""
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        # Zero previous gradients
        optimizer.zero_grad()
        # Run the model
        output = model(src, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        # Cut off the first column and reshape
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # Calculate and backprop loss
        loss = criterion(output, trg)
        loss.backward()

        # Clip gradients to prevent explosion
        clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    """Evaluate the model"""
    model.eval()
    epoch_loss = 0

    # No learning
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # No teacher forcing
            output = model(src, trg, 0)
            output_dim = output.shape[-1]

            # Cut off the first column and reshape
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # Calculate and accumulate loss
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = MAX_LEN):
    """Translate sentence after tokenization"""
    model.eval()
    # Tokenize
    if isinstance(sentence, str):
        tokens = sentence.split()
    else:
        tokens = [token for token in sentence]
    # Add delimiters
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    # Get indices of the words
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    # Encode the sentence tensor
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    # The first token
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        # Create tensor of the last target token
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        # Get the output
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)

        # Prediction
        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)
        # Loop till <eos>
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    # Convert to words
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    # Ignore begin of sentence token
    return trg_tokens[1:]

def get_ref_list(data,src):
    ref_list = []
    for e in data.examples:
        if e.src == list(src):
            ref_list.append(e.trg)
    return(ref_list)

def calculate_metrics(data, src_field, trg_field, model, device,dump_path = None,model_name='default', max_len = MAX_LEN):
    """Calculate BLEU and NIST metrics"""
    preds_file = dump_path+model_name+'-baseline-output.txt'
    refs_file = dump_path+model_name+'-devel-conc.txt'
    bleu_scores = []
    nist_scores = []
    multibleu_smoother = SmoothingFunction().method4
    refs = []
    preds = []

    # Group sources and references
    src_unique = list(set([tuple(e.src) for e in data.examples]))
    for src in src_unique:
        pred_trg = translate_sentence(src, src_field, trg_field, model, device, max_len)

        # Remove <eos>
        pred_trg = pred_trg[:-1]
        ref_list = get_ref_list(data,src) 

        preds.append(' '.join(pred_trg))
        refs.append(ref_list)
        
        # Calculate scores and save
        bleu_score = sentence_bleu(ref_list,pred_trg, smoothing_function=multibleu_smoother)
        nist_score = sentence_nist(ref_list,pred_trg, )       
        bleu_scores.append(bleu_score)
        nist_scores.append(nist_score)            

    # Dump all results in official e2e metric script compatible format
    file_refs = []
    for ref in refs:
        ref_sentences = [' '.join(tokens) for tokens in ref]
        refs_joined = '\n'.join(ref_sentences)
        file_refs.append(refs_joined)
    with open(preds_file,'w') as pred_f:
        pred_f.write('\n'.join(preds))  
    with open(refs_file,'w') as ref_f:
        ref_f.write('\n\n'.join(file_refs))  
    print (f'Writing files for {model_name}')
    print (f'Predictions in {preds_file}')
    print (f'References in {refs_file}')
    return np.mean(bleu_scores),np.mean(nist_scores)
