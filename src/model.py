#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Seq2Seq Model 

@author    : Reddy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random

seed = 7
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # Bidirectional GRU
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional = True)
        # Hidden dimension doubling due to bidirectional RNN
        self.fc = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):

        # Apply embedding to create dense vectors
        # src_len x batch -> src_len x batch x emb_dim
        embedded = self.dropout(self.embedding(src))

        # Pass embedding to bidirectional RNN
        # outputs: src_len x batch x hid*2
        # hidden: n_layers*2 x batch x hid
        outputs, hidden = self.rnn(embedded)

        # Get RNN states obtained by going through the input forward and backward
        last_fore_rnn = hidden[-2,:,:]
        last_back_rnn = hidden[-1,:,:]
        # Concatenate context vectors to create hidden: batch x hid
        final_hidden = torch.cat((last_fore_rnn, last_back_rnn), dim = 1)

        hidden = torch.tanh(self.fc(final_hidden))

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.weighted_energy_sum = nn.Linear(hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):

        src_len = encoder_outputs.shape[0]       

        # Repeat decoder hidden state to match src_len
        # hidden: batch x hid -> batch x 1 x hid -> batch x src_len x hid
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Make batch-first encoder outputs
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        stacked_by_hidden = torch.cat((hidden, encoder_outputs), dim = 2)

        # energy : batch x src_len x hid
        energy = torch.tanh(self.attn(stacked_by_hidden)) 
        
        attention = self.weighted_energy_sum(energy).squeeze(2) 

        # Return attention vector: batch x src_len (Every slice along src_len sums to 1)          
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim)
        
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
             
        # input : batch -> 1 x batch
        input = input.unsqueeze(0)
                
        # embedded : 1 x batch x hid
        embedded = self.dropout(self.embedding(input))
        
        # Apply attention to get a: batch x src_len       
        a = self.attention(hidden, encoder_outputs) 
        # a : batch x 1 x src_len                       
        a = a.unsqueeze(1)

        # Make batch first encoder outputs               
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # Batch matrix multiply encoder outputs with attention to get batch x 1 x hid*2
        weighted = torch.bmm(a, encoder_outputs)      
        # weighted : 1 x batch x hid*2  
        weighted = weighted.permute(1, 0, 2)
        
        # Concatenate embedded with weighted and send to RNN     
        # rnn_input: 1 x batch x hid*2 + emb_dim  
        rnn_input = torch.cat((embedded, weighted), dim = 2)
                    
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
                
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        concatenated_state = torch.cat((output, weighted, embedded), dim = 1)
        prediction = self.fc_out(concatenated_state)
        
        # Return prediction : batch x output_dim       
        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        trg_len = trg.shape[0]
        batch_size = src.shape[1]      
        trg_vocab_size = self.decoder.output_dim
        
        # Create a tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # Get all hidden states along with the final forward and backward ones
        encoder_outputs, hidden = self.encoder(src)
                
        # First input : <sos>
        input = trg[0,:]

        # Until we get the target sentence length, decode
        # Note: loop starts at 1 so that (<sos>,y,<eos>) corresponds to (0,y_hat,<eos>)
        for t in range(1, trg_len):
            
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            
            # Save the prediction for this token
            outputs[t] = output

            # Decide if teacher forcing is to be used
            teacher_force = random.random() < teacher_forcing_ratio     
                 
            # Next token to use is either from target or the prediction 
            input = trg[t] if teacher_force else output.argmax(1)

        return outputs