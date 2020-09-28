#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Train on processed e2e corpus
           
@author    : Reddy
"""
import random
import torch
import torch.nn as nn
from torch import optim

from torchtext.data import Field, BucketIterator, TabularDataset

from copy import deepcopy
import math
import sys
import argparse
import traceback
import pandas as pd


from model import Attention,Encoder,Decoder,Seq2Seq
from preprocess import write_corrupt_file
from utils import train,evaluate,translate_sentence,calculate_metrics

# Seed all randomness
seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Set device to run code on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(opts):
    """Main logic"""
    # The input is a path to the data files
    data_path = opts.infile
    e2e_train_dataset = 'new_trainset.csv'
    e2e_valid_dataset = 'new_devset.csv'
    e2e_test_dataset = 'new_testset.csv'

    # The output is the path to save model
    out_path = opts.outfile
    saved_model_best = out_path+'saved-model.pt'
    saved_model_least = out_path+'saved-model-least.pt'
    saved_model_last = out_path+'saved-model-last.pt'
    model_checkpoints = [saved_model_best,saved_model_least,saved_model_last]

    # Initialise data fields: MR, REF
    SRC = Field(init_token = '<sos>', eos_token = '<eos>')
    TRG = Field(init_token = '<sos>', eos_token = '<eos>')
    
    # Format of the data is: MR , REF
    data_fields = [('src',SRC),('trg', TRG)]
    
    # Split data
    train_data, valid_data, test_data = TabularDataset.splits(
        path=data_path, train=e2e_train_dataset,
        validation=e2e_valid_dataset, test=e2e_test_dataset, format='csv',
        fields=data_fields, skip_header=True)

    # Build vocabulary
    SRC.build_vocab(train_data)
    TRG.build_vocab(train_data)

    batch = 80

    # Create data loaders
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = batch,
    sort = False,
    repeat = False,
    sort_within_batch = True,
    sort_key = lambda x : len(x.src),
    device = device)

    # Training parameters
    num_epochs = 30
    clip_value = 10

    # Model parameters
    input_vocab_size = len(SRC.vocab)
    output_vocab_size = len(TRG.vocab)
    embedding_size = 512
    hidden_size = 512
    dropout = 0.1

    # Create the model
    attn = Attention(hidden_size)
    enc = Encoder(input_vocab_size, embedding_size, hidden_size, dropout)
    dec = Decoder(output_vocab_size, embedding_size, hidden_size, dropout, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    # Initialise model, optimizer and criterion
    optimizer = optim.Adam(model.parameters())
    # Schedule learning rate to decrease if loss starts increasing
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.5, verbose=True)
    trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)

    best_valid_loss = float('inf')
    valid_loss = best_valid_loss
    best_train_loss = float('inf')
    train_loss = best_train_loss
    default_value_dict = {  'epoch': 0,
                            'best_train_loss' : best_train_loss,
                            'best_valid_loss' : best_valid_loss,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'name' : 'default'
                         }
    to_save = {i: default_value_dict for i in model_checkpoints}
    base_teacher = 0.3
    decay = 0.9
    first_epoch=0
    # Check if resume training is turned on
    if opts.resume:
        checkpoint = torch.load(saved_model_last)
        model.load_state_dict(checkpoint['state_dict'])
        first_epoch = checkpoint.get('epoch',0)
        best_train_loss = checkpoint.get('best_train_loss',best_train_loss)
        best_valid_loss = checkpoint.get('best_valid_loss',best_valid_loss)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Resuming from epoch {first_epoch+1}')

    if not opts.eval:
        for epoch in range(first_epoch,num_epochs): 
            # Decay teacher forcing 
            teacher_forcing = base_teacher #* (decay**epoch) 

            train_loss = train(model, train_iterator, optimizer, criterion, clip_value, teacher_forcing)
            valid_loss = evaluate(model, valid_iterator, criterion)

            lr_scheduler.step(valid_loss)

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                # Save current model parameters
                to_save[saved_model_least] = {  'epoch': epoch + 1,
                                                'best_train_loss' : best_train_loss,
                                                'best_valid_loss' : best_valid_loss,
                                                'state_dict': deepcopy(model.state_dict()),
                                                'optimizer' : deepcopy(optimizer.state_dict()),
                                                'name' : 'least-train-loss'
                                            }
                            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # Save current model parameters
                to_save[saved_model_best] = {   'epoch': epoch + 1,
                                                'best_train_loss' : best_train_loss,
                                                'best_valid_loss' : best_valid_loss,
                                                'state_dict': deepcopy(model.state_dict()),
                                                'optimizer' : deepcopy(optimizer.state_dict()),
                                                'name' : 'least-val-loss'
                                            }

            print(f'Epoch: {epoch+1:02}')
            print(f'\tTrain Loss: {train_loss:.2f} | Train PPL: {math.exp(train_loss):5.2f}')
            print(f'\t Val. Loss: {valid_loss:.2f} |  Val. PPL: {math.exp(valid_loss):5.2f}')

            # Save the last model parameters
            to_save[saved_model_last] = {   'epoch': epoch + 1,
                                            'best_train_loss' : best_train_loss,
                                            'best_valid_loss' : best_valid_loss,
                                            'state_dict': deepcopy(model.state_dict()),
                                            'optimizer' : deepcopy(optimizer.state_dict()),
                                            'name' : 'last-epoch'
                                            }
        # Save models
        for name, value_dict in to_save.items():
            torch.save(value_dict,name)

    # Pick random indices in test data examples
    population = range(len(test_data.examples))
    samples =  random.sample(population, 3)   
    for point in model_checkpoints:
        checkpoint = torch.load(point)
        model.load_state_dict(checkpoint['state_dict'])
        saved_epoch = checkpoint['epoch']
        model_name = checkpoint['name']
        test_loss = evaluate(model, test_iterator, criterion)
        stats_string = f'Stats with the {model_name} model saved at epoch {saved_epoch}'
        stars = '*'*len(stats_string)
        print(stars+'\n'+stats_string+'\n'+stars)

        print(f'| Test Loss: {test_loss:.2f} | Test PPL: {math.exp(test_loss):5.2f} |')
        dump_path = opts.outfile
        bleu_score,nist_score = calculate_metrics(test_data, SRC, TRG, model, device, dump_path,model_name)
        print(f'BLEU score = {bleu_score:.2%} NIST score = {nist_score}')
        print('Example translations')
        population = range(len(test_data.examples))
        for i in samples:
            mr = vars(test_data.examples[i])['src']
            ref = vars(test_data.examples[i])['trg']

            print(f'MR = {mr}')
            print(f'REF = {ref}')
            translation = translate_sentence(mr, SRC, TRG, model, device)
            print(f'Predicted: {translation[:-1]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', metavar='FILE', dest='infile', required=True, default=None, help='Input data path with preprocessed files')
    parser.add_argument('-o', '--output', metavar='FILE', dest='outfile', required=True, default=None, help='Path to save model')
    parser.add_argument('-r', '--resume', dest='resume', action='store_true', help='Resume training from last saved checkpoint')
    parser.add_argument('-e', '--evaluate only', dest='eval', action='store_true', help='Evaluate only')

    opts = parser.parse_args(sys.argv[1:])

    try:
        main(opts)
    except Exception:
        print('Unhandled error!')
        traceback.print_exc()
        sys.exit(-1)

