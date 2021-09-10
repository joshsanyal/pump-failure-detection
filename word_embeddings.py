import pandas as pd
import numpy as np
from biobert_embedding import BiobertEmbedding
from gensim.models.callbacks import CallbackAny2Vec
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from Preprocessing import process, parse_impression
import os
from gensim.models import Word2Vec, KeyedVectors, FastText
import re
import string
from time import time
from nltk.tokenize import sent_tokenize, word_tokenize
from SentTokenizer import segment
from sklearn.pipeline import Pipeline
import random

def snippetLabel(s):
    keywords = pd.read_csv("dic/Dictionary.csv")
    keywords = keywords["Terms"]
    blocked_words = ["as directed", "back up", "backup", "company", "discuss", "verbalize", "scenario", "troubleshoot", "review", "event of", "case of", "plan"]
    for word in blocked_words:
        if (word in s.lower()):
            return 0
    for word in keywords:
        if (word.lower() + " failure" in s.lower() or word.lower() + " malfunction" in s.lower()):
            return 1
    return 0

class callback(CallbackAny2Vec):
    ### Callback to print loss after each epoch.

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        print('Time to train this epoch: {} mins'.format(round((time() - t) / 60, 2)))
        model.wv.save_word2vec_format('IP_w2v_epoch' + str(self.epoch + 1) + '.bin', binary=True)
        self.epoch += 1

def process(text):
    spaces = "                         "
    for i in range(22):
        text = text.replace(spaces[i:],"\n")
    newlines = "\n\n\n\n\n\n\n\n\n\n"
    for i in range(10):
        text = text.replace(newlines[i:],"\n")

    text = text.replace("\n", " ")
    text = re.sub('['+ string.punctuation + ']', ' ', text) #remove punctuation
    text = text.lower()

    words = text.split(' ')
    sent = []
    for j in range(len(words)):
        if (len(words[j]) > 2 and not words[j].isnumeric()):
            sent.append(words[j])
    return " ".join(sent)


sents = []
counter = 0
for gm_chunk in pd.read_csv("data/IP_progressnotes.csv", chunksize=10000, encoding='latin1'):
    print(counter)
    counter += 1
    df_relevant = gm_chunk.reset_index(drop = True)
    for i in (range(df_relevant.shape[0])):
        try: sents.append(process(df_relevant.iloc[i]['note_text']))
        except: ''''''
    if (counter == 4 ): break

t = time()
wv = Word2Vec(sents, size=300, window=30, min_count=50, sg=1, compute_loss=True, callbacks=[callback()], iter=10)
print('Time to build the model (50 epochs): {} mins'.format(round((time() - t) / 60, 2)))
wv.wv.save_word2vec_format('IP_w2v.bin', binary=True)

