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

def display_closestwords_tsnescatterplot(words, clusterSize):
    model = KeyedVectors.load_word2vec_format("/Users/josh/PycharmProjects/RecurrencePrediction/PubMed-w2v.bin", binary=True)

    arr = np.empty((0,200), dtype='f')
    word_labels = []

    for word in words:
        word_labels.append(word)
        # get close words
        close_words = model.similar_by_word(word,topn=35)

        # add the vector for each of the closest words to the array
        arr = np.append(arr, np.array([model[word]]), axis=0)
        counter = 0
        for wrd_score in close_words:
            if (wrd_score[0].islower() and "-" not in wrd_score[0] and "/" not in wrd_score[0] and "." not in wrd_score[0] and counter != clusterSize-1):
                counter += 1
                wrd_vector = model[wrd_score[0]]
                word_labels.append(wrd_score[0])
                arr = np.append(arr, np.array([wrd_vector]), axis=0)
        print(word_labels)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=2554)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot

    colors = ["lightseagreen", "forestgreen", "mediumvioletred", "coral", "sienna", "red"]
    for i in range(len(words)):
        plt.scatter(x_coords[i*clusterSize:(i+1)*clusterSize], y_coords[i*clusterSize:(i+1)*clusterSize], color = colors[i], s = 50, label = words[i] + "-related terms")

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, color="black", xy=(x, y), xytext=(2, 2), size = 14, textcoords='offset points')
    plt.legend()
    plt.xlim(x_coords.min()-10, x_coords.max()+10)
    plt.ylim(y_coords.min()-10, y_coords.max()+10)
    plt.show()

display_closestwords_tsnescatterplot(["metformin", "insulin", "glucose", "creatnine", "malfunction", "diabetes"], 6)

def display_documents_tsnescatterplot():
    dataset = pd.read_excel("data/IP_dataset_reduced.xlsx")
    numFalse = 0
    numTrue = 0
    order = random.sample(range(len(dataset)), 3121)
    used = []
    patient_id = []
    for i in order:
        if (dataset["label"][i] == 0 and numFalse < 50):
            used.append(i)
            numFalse += 1
            patient_id.append(dataset["person_id"][i])
        elif (dataset["label"][i] == 1 and numTrue < 120):
            numTrue += 1
            used.append(i)
            patient_id.append(dataset["person_id"][i])

    snippets = []
    indices = [[],[]]
    snippets_df = pd.read_csv("data/IP_snippets_dataset.csv")
    blocked_words = ["directed", "back up", "backup", "company", "discuss", "verbalize", "scenario", "troubleshoot", "review", "event", "case", "plan","no pump failure", "no pump malfunction", "no failure", "no malfunction"]
    for i in range(len(snippets_df)):
        if (i % 1000 == 0): print(i)
        if snippets_df['person_id'][i] in patient_id:
            try:
                progress = True
                originalTxt = "".join(process(snippets_df['snippet'][i]))
                for term in blocked_words:
                    if re.search(r'\b' + term.lower() + r'\b', originalTxt):
                        progress = False
                if progress:
                    snippets.append(originalTxt)
                    indices[snippetLabel(originalTxt)].append(len(snippets)-1)
            except: ''''''

    biobert = BiobertEmbedding()
    x = []
    for i, snip in enumerate(snippets):
        if (len(snip) > 1000): snip = snip[0:1000]
        x.append(np.array(biobert.sentence_vector(snip)))

    '''
    clf = Pipeline([('vect', CountVectorizer(max_features=300)),('tfidf', TfidfTransformer()),])
    clf.fit_transform(snippets)
    x = np.empty([len(snippets),300])
    for i, snip in enumerate(snippets):
        x[i] = clf.transform([snip]).toarray()
    '''

    '''
    tfidf = TfidfVectorizer(sublinear_tf=True)
    features = tfidf.fit_transform(snippets)
    dict = {val : idx for idx, val in enumerate(tfidf.get_feature_names())}

    preTrainedPath = "IP_MIMIC_w2v.bin"
    wv = KeyedVectors.load_word2vec_format(preTrainedPath, binary=True)

    x = np.empty([len(snippets),300])
    i = 0
    for report in snippets:
        words = report.split()
        avgFeat = np.zeros(300)
        for word in words:
            try: # if in vocab
                vector = np.multiply(wv[word],features[i,dict[word]])
                avgFeat = np.add(avgFeat, vector)
            except: # if out of vocab ''''''
        if (len(words) == 0): x[i] = avgFeat
        else: x[i] = avgFeat/len(words)
        i += 1
    '''

    tsne = TSNE(n_components=2, random_state=76)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(x)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]


    plt.rcParams.update({'font.size': 16})

    plt.scatter([x_coords[index] for index in indices[0]], [y_coords[index] for index in indices[0]], color = "blue", label = "Negative Failure Status")
    plt.scatter([x_coords[index] for index in indices[1]], [y_coords[index] for index in indices[1]], color = "red", label = "Positive Failure Status")
    plt.legend()
    plt.show()


display_documents_tsnescatterplot()


