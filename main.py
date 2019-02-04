import pandas as pd
import torch
import numpy as np
import re
from bs4 import BeautifulSoup

from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string


def clean_str(string, max_seq_len):
    """
    adapted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = BeautifulSoup(string, "lxml").text
    string = re.sub(r"[^A-Za-z0-9(),!?\"\`]", " ", string)
    string = re.sub(r"\"s", " \"s", string)
    string = re.sub(r"\"ve", " \"ve", string)
    string = re.sub(r"n\"t", " n\"t", string)
    string = re.sub(r"\"re", " \"re", string)
    string = re.sub(r"\"d", " \"d", string)
    string = re.sub(r"\"ll", " \"ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    s = string.strip().lower().split(" ")
    if len(s) > max_seq_len:
        return s[0:max_seq_len]
    return s


## creates a 3D list of format paragraph[sentence[word]]
def create3DList(df, col, max_sent_len, max_seq_len):
    x = []
    for docs in df[col].as_matrix():
        x1 = []
        idx = 0
        for seq in "|||".join(re.split("[.?!]", docs)).split("|||"):
            x1.append(clean_str(seq, max_sent_len))
            if (idx >= max_seq_len - 1):
                break
            idx = idx + 1
        x.append(x1)
    return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('yelp.csv')
    print(df.head())

    ## mark the columns which contains text for classification and target class
    col_text = 'text'
    col_target = 'cool'

    cls_arr = np.sort(df[col_target].unique()).tolist()
    classes = len(cls_arr)

    cls_arr = np.sort(df[col_target].unique()).tolist()
    classes = len(cls_arr)

    ## divide dataset in 80% train 10% validation 10% test as done in the paper
    length = df.shape[0]
    train_len = int(0.8 * length)
    val_len = int(0.1 * length)
    print(length)

    train = df[:train_len]
    val = df[train_len:train_len + val_len]
    test = df[train_len + val_len:]

    ## Fix the maximum length of sentences in a paragraph and words in a sentence
    max_sent_len = 12
    max_seq_len = 25

    ## divides review in sentences and sentences into word creating a 3DList
    x_train = create3DList(train, col_text, max_sent_len, max_seq_len)
    x_val = create3DList(val, col_text, max_sent_len, max_seq_len)
    x_test = create3DList(test, col_text, max_sent_len, max_seq_len)
    print("x_train: {}".format(len(x_train)))
    print("x_val: {}".format(len(x_val)))
    print("x_test: {}".format(len(x_test)))

    stoplist = stopwords.words('english') + list(string.punctuation)
    stemmer = SnowballStemmer('english')
    x_train_texts = [[[stemmer.stem(word.lower()) for word in sent if word not in stoplist] for sent in para]
                     for para in x_train]
    x_test_texts = [[[stemmer.stem(word.lower()) for word in sent if word not in stoplist] for sent in para]
                    for para in x_test]
    x_val_texts = [[[stemmer.stem(word.lower()) for word in sent if word not in stoplist] for sent in para]
                   for para in x_val]

    ## calculate frequency of words
    from collections import defaultdict

    frequency1 = defaultdict(int)
    for texts in x_train_texts:
        for text in texts:
            for token in text:
                frequency1[token] += 1
    for texts in x_test_texts:
        for text in texts:
            for token in text:
                frequency1[token] += 1
    for texts in x_val_texts:
        for text in texts:
            for token in text:
                frequency1[token] += 1

    ## remove  words with frequency less than 5.
    x_train_texts = [[[token for token in text if frequency1[token] > 5]
                      for text in texts] for texts in x_train_texts]

    x_test_texts = [[[token for token in text if frequency1[token] > 5]
                     for text in texts] for texts in x_test_texts]
    x_val_texts = [[[token for token in text if frequency1[token] > 5]
                    for text in texts] for texts in x_val_texts]
