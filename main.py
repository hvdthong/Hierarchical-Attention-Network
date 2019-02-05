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
import more_itertools

## calculate frequency of words
from collections import defaultdict


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


## Make the the multiple attention with word vectors.
def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i]
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if (attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)


## The word RNN model for generating a sentence vector
class WordRNN(nn.Module):
    def __init__(self, vocab_size, embedsize, batch_size, hid_size):
        super(WordRNN, self).__init__()
        self.batch_size = batch_size
        self.embedsize = embedsize
        self.hid_size = hid_size
        ## Word Encoder
        self.embed = nn.Embedding(vocab_size, embedsize)
        self.wordRNN = nn.GRU(embedsize, hid_size, bidirectional=True)
        ## Word Attention
        self.wordattn = nn.Linear(2 * hid_size, 2 * hid_size)
        self.attn_combine = nn.Linear(2 * hid_size, 2 * hid_size, bias=False)

    def forward(self, inp, hid_state):
        emb_out = self.embed(inp)

        out_state, hid_state = self.wordRNN(emb_out, hid_state)

        word_annotation = self.wordattn(out_state)
        attn = F.softmax(self.attn_combine(word_annotation), dim=1)

        sent = attention_mul(out_state, attn)
        return sent, hid_state


## The HAN model
class SentenceRNN(nn.Module):
    def __init__(self, vocab_size, embedsize, batch_size, hid_size, c):
        super(SentenceRNN, self).__init__()
        self.batch_size = batch_size
        self.embedsize = embedsize
        self.hid_size = hid_size
        self.cls = c
        self.wordRNN = WordRNN(vocab_size, embedsize, batch_size, hid_size)
        ## Sentence Encoder
        self.sentRNN = nn.GRU(embedsize, hid_size, bidirectional=True)
        ## Sentence Attention
        self.sentattn = nn.Linear(2 * hid_size, 2 * hid_size)
        self.attn_combine = nn.Linear(2 * hid_size, 2 * hid_size, bias=False)
        self.doc_linear = nn.Linear(2 * hid_size, c)

    def forward(self, inp, hid_state_sent, hid_state_word):
        s = None
        ## Generating sentence vector through WordRNN
        for i in range(len(inp[0])):
            r = None
            for j in range(len(inp)):
                if (r is None):
                    r = [inp[j][i]]
                else:
                    r.append(inp[j][i])
            r1 = np.asarray([sub_list + [0] * (max_seq_len - len(sub_list)) for sub_list in r])
            _s, state_word = self.wordRNN(torch.cuda.LongTensor(r1).view(-1, batch_size), hid_state_word)
            if (s is None):
                s = _s
            else:
                s = torch.cat((s, _s), 0)

                out_state, hid_state = self.sentRNN(s, hid_state_sent)
        sent_annotation = self.sentattn(out_state)
        attn = F.softmax(self.attn_combine(sent_annotation), dim=1)

        doc = attention_mul(out_state, attn)
        d = self.doc_linear(doc)
        cls = F.log_softmax(d.view(-1, self.cls), dim=1)
        return cls, hid_state

    def init_hidden_sent(self):
        return Variable(torch.zeros(2, self.batch_size, self.hid_size))

    def init_hidden_word(self):
        return Variable(torch.zeros(2, self.batch_size, self.hid_size))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('yelp.csv')
    print(df.head())

    ## mark the columns which contains text for classification and target class
    col_text = 'text'
    col_target = 'cool'

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
    max_sent_len = 12  # max length of a sentence
    max_seq_len = 25  # max length of sentences in a paragraph

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
    texts = list(more_itertools.collapse(x_train_texts[:] + x_test_texts[:] + x_val_texts[:], levels=1))

    ## train word2vec model on all the words
    word2vec = Word2Vec(texts, size=200, min_count=5)
    word2vec.save("dictonary_yelp")

    ## convert 3D text list to 3D list of index
    x_train_vec = [[[word2vec.wv.vocab[token].index for token in text]
                    for text in texts] for texts in x_train_texts]
    x_test_vec = [[[word2vec.wv.vocab[token].index for token in text]
                   for text in texts] for texts in x_test_texts]
    x_val_vec = [[[word2vec.wv.vocab[token].index for token in text]
                  for text in texts] for texts in x_val_texts]
    weights = torch.FloatTensor(word2vec.wv.syn0)
    print('Shape of word vectors: ', weights.shape)

    vocab_size = len(word2vec.wv.vocab)

    y_train = train[col_target].tolist()
    y_test = test[col_target].tolist()
    y_val = val[col_target].tolist()

    ## converting list to tensor
    y_train_tensor = [torch.FloatTensor([cls_arr.index(label)]) for label in y_train]
    y_val_tensor = [torch.FloatTensor([cls_arr.index(label)]) for label in y_val]
    y_test_tensor = [torch.FloatTensor([cls_arr.index(label)]) for label in y_test]

    max_seq_len = max([len(seq) for seq in itertools.chain.from_iterable(x_train_vec + x_val_vec + x_test_vec)])
    max_sent_len = max([len(sent) for sent in (x_train_vec + x_val_vec + x_test_vec)])
