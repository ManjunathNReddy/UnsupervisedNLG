#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Read and clean "e2e" corpus and create custom train, valid and test sets

@author    : Reddy
"""

import argparse
import sys
import traceback

import pandas as pd
import numpy as np

import unicodedata
import string
import re
import random

seed = 7
random.seed(seed)


# Sentence length constraints
MIN_LEN = 2
MAX_LEN = 60

# Turn a Unicode string to plain ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Trim and remove non-letter and non-number characters
def normalize_string(s):
    s = unicode_to_ascii(s.strip())
    s = re.sub(r"[^0-9a-zA-Z-.]+", r" ", s)
    s = s.strip()
    return s

def clean_data_frame(df):
    df = df.applymap(normalize_string)
    df_boolean_mask = df.applymap(lambda x: MIN_LEN <= len(x.split()) <= MAX_LEN)
    df_boolean_mask = df_boolean_mask.iloc[:,0] & df_boolean_mask.iloc[:,1]
    # Remove too short and too long sentences
    return df[df_boolean_mask]

def custom_shuffle(correct,corrupt, n=2):
  # Generate all n-grams
  ngrams = list(zip(*[correct[i:] for i in range(n)]))

  corc = corrupt.copy()

  # Create a list of n-grams and tokens to shuffle

  # Add the first token
  to_shuffle = [[corc.pop(0)]]

  while corc:     
      # Only one token left - add it to shuffle
      if len(corc) == 1:
          to_shuffle.append([corc.pop()])      
      else:
        # Form a candidate n-gram combining the tail of shuffle list with the head of corrupt token list
        corrupt_token_head = corc[0]
        shuffle_tail = to_shuffle[-1][-1]
        candidate_ngram = (shuffle_tail, corrupt_token_head)
        # If the candidate bigram is not present in the ngram list, add it as 1-gram in the list 
        if candidate_ngram not in ngrams:
          to_shuffle.append([corc.pop(0)])
        # Else, we found a bigram in the original list - so, save it
        else:
          to_shuffle[-1].append(corc.pop(0))
  # Shuffle all n-grams
  random.shuffle(to_shuffle)
  # Flatten the list
  shuffled = [elem for lst in to_shuffle for elem in lst]
  return shuffled

def get_training_pairs(sentence_tokenset,most_freq_set):
  """ Return a zip of pairs of X(corrupt sentence) and Y(correct sentence) """
  # Probability of keeping a frequent word
  p_keep=0.4
  correct_corpus = []
  corrupt_corpus = []
  for token_list in sentence_tokenset:
    corrupt_sentence = []
    for word in token_list:
      # Check if the word is not in the most frequent list
      # If it's frequent, keep it with a predefined probability
      if word not in most_freq_set or np.random.rand() < p_keep:
        corrupt_sentence.append(word)

    # It is possible that the corrupt sentence is empty
    # In this case, sample the original sentence til a random length
    if not corrupt_sentence:
      length_choice = random.randint(1,len(token_list))
      corrupt_sentence = token_list[:length_choice]

    # Shuffle corrupt sentence except the bigrams
    corrupt_sentence = custom_shuffle(correct=token_list,corrupt=corrupt_sentence)

    corrupt_corpus.append(" ".join(corrupt_sentence))
    correct_corpus.append(" ".join(token_list))
  
  return zip(corrupt_corpus,correct_corpus)

def write_corrupt_file(corpus,columns,file_path):
    # Tokenise the text
    tokens = []
    sentence_tokenset = []
    for line in corpus:
        sentence_tokens = line.split()
        sentence_tokenset.append(sentence_tokens)
        tokens.extend(sentence_tokens)
    words, counts = np.unique(tokens, return_counts=True)
    high_freq_words = counts>100
    most_freq_set = set(words[high_freq_words])
    training_pairs = get_training_pairs(sentence_tokenset, most_freq_set) 

    trainset = pd.DataFrame.from_records(training_pairs,columns = columns)

    trainset.to_csv(file_path, index=False)
    return len(trainset)


def get_value_sentence(text):
    """Preprocess e2e meaning representation into slots' values representation"""
    # Get feature names
    raw_features = re.findall(r"[\w|\s]+\[", text)
    # Remove brackets and spaces
    features = [re.sub("\[|\s", "", feature) for feature in raw_features]

    # Get values
    raw_values = re.findall(r"\[[\w|\s|£|-]+\]", text)
    # Remove brackets
    values = [re.sub("\[|\]", "", value) for value in raw_values]

    tokens = []
    for feat,value in zip(features,values):
        # Replace boolean feature
        if unicode_to_ascii(feat) == 'familyFriendly':
            val = 'family friendly' if unicode_to_ascii(value) == 'yes' else 'not family friendly'
            tokens.extend(val)
        else:
            tokens.extend(value)
        tokens.extend(' ')
    
    return ''.join(tokens)

def create_val_and_test_sets(path_prefix,old_val_file,new_val_file,new_test_file, is_unsuper):
    """Remove invalid sentences in the old_val_file and split it to new_val_file & new_test_file"""
    old_val = path_prefix+old_val_file
    new_val = path_prefix+new_val_file
    new_test = path_prefix+new_test_file

    valset = pd.read_csv(old_val)
    if is_unsuper:
        valset['mr'] = valset['mr'].apply(get_value_sentence)        
    valset = clean_data_frame(valset)

    # Get groups of meaning representations
    mr_groups = valset.groupby(['mr'])
    group_list = [mr_groups.get_group(x) for x in mr_groups.groups]

    # Split into val-test
    test_group = pd.concat(group_list[:279])
    val_group = pd.concat(group_list[279:])

    # Save the results in new files
    val_group.to_csv(new_val, index=False)
    test_group.to_csv(new_test, index=False)

    return len(val_group),len(test_group)


def main(opts):
    """Main logic"""
    # E2E specific files
    e2e_train_dataset = "trainset.csv"
    e2e_new_train_dataset = "new_trainset.csv"
    e2e_valid_dataset = "devset.csv"
    e2e_new_valid_dataset = "new_devset.csv"
    #e2e_test_dataset = "testset_w_refs.csv"
    e2e_new_test_dataset="new_testset.csv"
    e2e_unsuper_dataset = "news-commentary-v15.en"
    is_unsuper = opts.unsuper
    is_ood = opts.ood

    # Read corpus path
    corpus_path = opts.infile

    # Create train set
    old_train_file = corpus_path+e2e_train_dataset
    new_train_file = corpus_path+e2e_new_train_dataset
    trainset = pd.read_csv(old_train_file)
    trainset = clean_data_frame(trainset)  
    train_len = len(trainset)
    columns = ['mr','ref']

    # Create val and test sets
    val_len, test_len = create_val_and_test_sets(corpus_path, e2e_valid_dataset, e2e_new_valid_dataset, e2e_new_test_dataset, (is_unsuper or is_ood))

    if is_unsuper:
        if is_ood:
            ood_train = corpus_path+e2e_unsuper_dataset
            # Clean dataset
            dataset = pd.read_csv(ood_train,sep='\t',header=None)
            # Duplicate column
            dataset['ref'] = dataset.iloc[:,0]
            dataset = clean_data_frame(dataset)
            dataset.columns = columns
            # Append ood data
            trainset = trainset.append(dataset, ignore_index=True)
        # Get the words and their corresponding counts of vocabulary
        corpus = trainset['ref'].values.tolist()
        
        train_len = write_corrupt_file(corpus,columns,new_train_file)
    else:
        train_len = len(trainset)
        trainset.to_csv(new_train_file, index=False)

    print(f'Created new train ({train_len}), dev({val_len}), test({test_len}) files at {corpus_path}!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', metavar='PATH', dest='infile', required=True, default=None, help='Path to e2e directory(including trailing slash)')
    parser.add_argument('-u', '--unsuper', dest='unsuper', action='store_true', help='Basic Unsupervised setup')
    parser.add_argument('-o', '--ood', dest='ood', action='store_true', help='with Out-of-domain data (Unsupervised setup)')

    opts = parser.parse_args(sys.argv[1:])

    try:
        main(opts)
    except Exception:
        print('Unhandled error!')
        traceback.print_exc()
        sys.exit(-1)

