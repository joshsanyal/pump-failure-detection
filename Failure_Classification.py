import pandas as pd
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from Preprocessing import process, text2embed
from xgboost import XGBClassifier
import numpy as np
from Preprocessing import parse_impression
from datetime import datetime, timedelta
from nltk.tag import pos_tag
from sklearn.linear_model import LogisticRegression
from numpy import zeros
import pickle
import scipy
from time import time
import statistics
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import tensorflow as tf
import re
import string
import random
from random import shuffle
from biobert_embedding.embedding import BiobertEmbedding
import torch
from textaugment import Wordnet
from textaugment import Translate
from textaugment import EDA
from textaugment import Word2vec
import nltk
from transformers import BertForSequenceClassification
import joblib
from sklearn import datasets, metrics, model_selection, svm

def snippetLabel(s):
    keywords = pd.read_csv("dic/Dictionary.csv")
    keywords = keywords["Terms"]
    blocked_words = ["directed", "backup", "company", "instructions", "discuss", "verbalize", "scenario", "troubleshoot", "review", "event", "case", "plan"]
    for word in blocked_words:
        if (word in s.lower()):
            return 0
    for word in keywords:
        if ("pump failure" in s.lower() or "pump malfunction" in s.lower()):
            return 1
    return 0


### DATA PREP
# Convert string to datetime object
def str2Date(x):
    return datetime.strptime(x, '%Y-%m-%d')

# Dictionary to keep track of positive labels
failures = pd.read_csv("data/IP_faliure.csv")
failure_dict = {}
for i in range(failures.shape[0]):
    failure_dict[failures['person_id'][i]] = failures['condition_start_DATE'][i]

# DICTIONARIES FOR THE NOTES
snippets_dict = {}
notes_dict = {}
start_date = {}

# KEYWORD
keywords = pd.read_csv("dic/Dictionary.csv")
keywords = keywords["Terms"]

# GENERATE SNIPPETS
def snippets(rawnote):
    rawnote = rawnote.split('PLAN:')[0]
    str1 = rawnote
    str1 = ' '+str1+' '
    snippets = []
    str1 = re.sub(r'[^\x00-\x7f]',r' ', str1)
    re1='(\\d+)'	# Integer Number 1
    re2='(\\.)'	# Any Single Character 1
    re3='( )'	# White Space 1
    re4='((?:[a-z][a-z]+))'	# Word 1
    rg = re.compile(re1+re2+re3+re4,re.IGNORECASE|re.DOTALL)
    str1 = re.sub(rg,r'\n', str1)
    str1 = re.sub('\s\s\s+','\n',str1)
    str1 = str1.replace('\r', '\n')
    str1 = str1.replace('#', '\n')
    paragraphs = [p for p in str1.split('\n') if p]
    for paragraph in paragraphs:
        temp_sentences = sent_tokenize(paragraph)
        for i in range(len(temp_sentences)):
            temp_sentences[i] = temp_sentences[i].lower()
            for term in keywords:
                if re.search(r'\b' + term.lower() + r'\b', temp_sentences[i]):
                    snippets.append(temp_sentences[i])
    return " ".join(snippets)

# ENTER NOTES INTO THE DICTIONARIES
def dictEntry(snippets, note, i, new):
    if new:
        snippets_dict[df_relevant.iloc[i]['person_id']] = snippets
        notes_dict[df_relevant.iloc[i]['person_id']] = note
    elif (len(snippets) > 2):
        if (len(snippets_dict[df_relevant.iloc[i]['person_id']]) > 2): snippets_dict[df_relevant.iloc[i]['person_id']] += " || "
        if (len(notes_dict[df_relevant.iloc[i]['person_id']]) > 2): notes_dict[df_relevant.iloc[i]['person_id']] += " || "
        snippets_dict[df_relevant.iloc[i]['person_id']] += snippets
        notes_dict[df_relevant.iloc[i]['person_id']] += note

counter = 0
weakneg = 0
failureCount = 0
for gm_chunk in pd.read_csv("data/IP_progressnotes.csv", chunksize=10000, encoding='latin1'):
    print(counter)
    counter += 1
    df_relevant = gm_chunk.reset_index(drop = True)
    for i in (range(df_relevant.shape[0])):
        try:
            # POSITIVE LABEL
            if (df_relevant.iloc[i]['person_id'] in failure_dict):
                notedate = str2Date(df_relevant.iloc[i]['note_DATE'])
                failuredate = str2Date(failure_dict[df_relevant.iloc[i]['person_id']])
                if (notedate + timedelta(days=35) >= failuredate and failuredate + timedelta(days=35) >= notedate):
                    dictEntry(snippets(df_relevant.iloc[i]['note_text']), df_relevant.iloc[i]['note_text'], i, df_relevant.iloc[i]['person_id'] not in snippets_dict)
            # NO LABEL
            else:
                # ALREADY SELECTED
                if (df_relevant.iloc[i]['person_id'] in start_date):
                    notedate = str2Date(df_relevant.iloc[i]['note_DATE'])
                    startdate = str2Date(start_date[df_relevant.iloc[i]['person_id']])
                    if (startdate + timedelta(days=64) >= notedate and notedate >= startdate):
                        dictEntry(snippets(df_relevant.iloc[i]['note_text']), df_relevant.iloc[i]['note_text'], i, False)
                # IF POSITIVE
                elif("pump failure" in df_relevant.iloc[i]['note_text'].lower() or "malfunction" in df_relevant.iloc[i]['note_text'].lower()):
                    failureCount += 1
                    start_date[df_relevant.iloc[i]['person_id']] = df_relevant.iloc[i]['note_DATE']
                    dictEntry(snippets(df_relevant.iloc[i]['note_text']), df_relevant.iloc[i]['note_text'], i, True)
                # IF WE NEED MORE NEGATIVE
                elif(weakneg < 3000):
                    weakneg += 1
                    start_date[df_relevant.iloc[i]['person_id']] = df_relevant.iloc[i]['note_DATE']
                    dictEntry(snippets(df_relevant.iloc[i]['note_text']), df_relevant.iloc[i]['note_text'], i, True)

        except: ''''''

# Creating the dataframe
column_names = ["person_id", "notes", "snippets", 'label']
combined_df = pd.DataFrame(columns = column_names)
index = 0

for person_id in snippets_dict:
    try: 
        if (len(snippets_dict[person_id]) > 2 and person_id % 1 == 0):
            label = np.NaN
            if person_id in failure_dict: label = 1
            combined_df.loc[index] = [person_id, notes_dict[person_id], snippets_dict[person_id], label]
            index += 1
    except: ''''''

combined_df.to_excel("data/InsulinPumpSnippets.xlsx")


#REDUCTION
id_dict = {}
dataset = pd.read_csv("data/IP_dataset_reduced.csv")
for i in range(len(dataset)):
    id_dict[dataset["person_id"][i]] = dataset["person_id"][i]

df = pd.read_excel("data/InsulinPumpSnippets.xlsx")
column_names = ["person_id", "notes", "snippets", 'label']
combined_df = pd.DataFrame(columns = column_names)
index = 0
for i in range(len(df)):
    if (df['person_id'][i] in id_dict):
        combined_df.loc[index] = [df['person_id'][i], df['notes'][i], df['snippets'][i], df['label'][i]]
        index += 1
combined_df.to_excel("data/IP_dataset_reduced.xlsx")

#SNIPPETS EXTRACTION FOR TRAINING
keywords = pd.read_csv("dic/Dictionary.csv")
keywords = keywords["Terms"]

df = pd.read_excel("data/InsulinPumpSnippets.xlsx")

snippets_df = pd.DataFrame(columns=['person_id',"snippet","label"])
personid = []
snippets = []
labels = []

print(df.shape[0])
for i in range(df.shape[0]):
    if (i % 100 == 0): print(i)
    snip = sent_tokenize(df['snippets'][i])
    for s in snip:
        label = 0
        snippets.append(s)
        personid.append(df['person_id'][i])
        if ("company" not in s.lower() and "review" not in s.lower() and "event of" not in s.lower() and "case of" not in s.lower() and "plan" not in s.lower()):
            for word in keywords:
                if (word.lower() + " failure" in s.lower() or word.lower() + " malfunction" in s.lower()):
                    label = 1
        labels.append(label)

for i in range(len(labels)):
    snippets_df.loc[i] = [personid[i], snippets[i], labels[i]]
snippets_df.to_csv("data/IP_snippets_dataset.csv", index=False)


# RANDOMIZATION
#RANDOMIZATION STUFF
for i in order:
    if (dataset["label"][i] == 0 and numFalse < 400):
        used.append(i)
        numFalse += 1
        patient_id.append(dataset["person_id"][i])
    elif (dataset["label"][i] == 1 and numTrue < 63):
        numTrue += 1
        used.append(i)
        patient_id.append(dataset["person_id"][i])

 #FIXED TRAINING/TEST
ds = pd.read_excel("data/IP_dataset_TFIDFsnippet_classified_augment.xlsx")
for i in range(len(ds)):
    if (np.isnan(ds["CLASSIFICATION"][i])):
        patient_id.append(dataset["person_id"][i])
        used.append(i)


##### USING SNIPPETS
dataset = pd.read_excel("data/IP_dataset_reduced.xlsx")

numFalse = 0
numTrue = 0
counter = 0

order = random.sample(range(len(dataset)), 3121)
used = []
patient_id = []

for i in order:
    if (dataset["label"][i] == 0 and numFalse < 700):
        used.append(i)
        numFalse += 1
        patient_id.append(dataset["person_id"][i])
    elif (dataset["label"][i] == 1 and numTrue < 90):
        numTrue += 1
        used.append(i)
        patient_id.append(dataset["person_id"][i])

x, y = [], []

t = Wordnet(v=True, n=True, p = 0.1)
augmentDel = EDA()
snippets_df = pd.read_csv("data/IP_snippets_dataset.csv")
for i in range(len(snippets_df)):
    if (i % 1000 == 0): print(i)
    if snippets_df['person_id'][i] in patient_id:
        try:
            originalTxt = " ".join(process(snippets_df['snippet'][i]))
            progress = True
            blocked_words = ["directed", "back up", "backup", "company", "discuss", "verbalize", "scenario", "troubleshoot", "review", "event", "case", "plan","no pump failure", "no pump malfunction", "no failure", "no malfunction"]
            for term in blocked_words:
                if re.search(r'\b' + term.lower() + r'\b', originalTxt):
                    progress = False
            if progress:
                x.append(originalTxt)
                y.append(snippetLabel(originalTxt))
                ''''''
                if (snippetLabel(originalTxt) == 1):
                    for i in range(10):
                        txt = t.augment(originalTxt)
                        x.append(txt)
                        y.append(snippetLabel(txt))
                        txt = augmentDel.random_deletion(txt, p=0.25)
                        x.append(txt)
                        y.append(snippetLabel(txt))
        except: ''''''

print(len(y)-sum(y))
print(sum(y))

class_weight = []
for label in y:
    if label == 0: class_weight.append(len(y)/(len(y)-sum(y)))
    else: class_weight.append(len(y)/sum(y))

text_clf = Pipeline([('vect', CountVectorizer(max_features=300)),('tfidf', TfidfTransformer()),
                     ('clf', XGBClassifier(max_depth = 6, n_estimators= 100, booster = "gbtree", reg_alpha = 0.2, )),])
text_clf.fit(x,y)
#joblib.dump(text_clf, 'data/model.pkl')

preds = []
selected_snip = []

pred = []
testY = []
for i in range(len(dataset)):
    label = 0
    selected = ""
    if i in used:
        preds.append(np.NaN)
        selected_snip.append(selected)
    else:
        sents = sent_tokenize(dataset['snippets'][i])
        for sent in sents:
            try:
                txt = " ".join(process(sent))
                #blocked_words = ["directed", "backup", "company", "instructions", "discuss", "verbalize", "scenario", "troubleshoot", "review", "event", "case", "plan"]
                prob = text_clf.predict_proba([txt])
                #for word in blocked_words:
                #    if word in txt: prob[0][1] = 0

                if (prob[0][1] > 0):
                    pred.append(prob[0][1])
                    testY.append(snippetLabel(txt))

                if (prob[0][1] > label):
                    selected = sent
                    label = prob[0][1]
            except: ''''''
        preds.append(label)
        selected_snip.append(selected)

dataset["selected_snippet"] = selected_snip
dataset["CLASSIFICATION"] = preds
#dataset.to_excel("data/IP_dataset_TFIDFsnippet_augmented.xlsx")


pred = []
testY = []
for i in range(len(dataset)):
    if (not np.isnan(dataset["CLASSIFICATION"][i])):
        pred.append(dataset["CLASSIFICATION"][i])
        testY.append(dataset["label"][i])

fpr, tpr, _ = metrics.roc_curve(testY, pred)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='cornflowerblue', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

##### WHOLE NOTE
dataset = pd.read_excel("data/IP_dataset_reduced.xlsx")

used = []
patient_id = []

ds = pd.read_excel("data/IP_dataset_TFIDFsnippet_classified_augment.xlsx")
for i in range(len(ds)):
    if (np.isnan(ds["CLASSIFICATION"][i])):
        patient_id.append(dataset["person_id"][i])
        used.append(i)

x, y = [], []

t = Wordnet()
augmentDel = EDA()
for i in range(len(dataset)):
    if (i % 1000 == 0): print(i)
    if dataset['person_id'][i] in patient_id:
        try:
            x.append(" ".join(process(dataset['snippets'][i])))
            y.append(dataset['label'][i])
            if (dataset['label'][i] == 1):
                txt = augmentDel.random_deletion(" ".join(process(dataset['snippets'][i])), p=0.1)
                x.append(txt)
                y.append(1)
                for i in range(10):
                    txt = " ".join(process(t.augment(dataset['snippets'][i],  p = 0.2)))
                    x.append(txt)
                    y.append(1)
                    txt = augmentDel.random_deletion(txt, p=0.1)
                    x.append(txt)
                    y.append(1)
        except: ''''''

print(sum(y))
print(len(y))

text_clf = Pipeline([('vect', CountVectorizer(max_features=300)),('tfidf', TfidfTransformer()),
                     ('clf', XGBClassifier(max_depth = 6, n_estimators= 100, booster = "gbtree", reg_alpha = 0.2, )),])
text_clf.fit(x,y)

preds = []
for i in range(len(dataset)):
    if i in used:
        preds.append(np.NaN)
    else:
        prob = text_clf.predict_proba([" ".join(process(dataset['snippets'][i]))])
        preds.append(prob[0][1])

dataset["CLASSIFICATION"] = preds
#dataset.to_excel("data/IP_dataset_TFIDF_classified_augmented.xlsx")


pred = []
testY = []
#dataset = pd.read_excel("data/IP_dataset_TFIDFsnippet_classified_augment.xlsx")
for i in range(len(dataset)):
    if (not np.isnan(dataset["CLASSIFICATION"][i])):
        pred.append(dataset["CLASSIFICATION"][i])
        testY.append(dataset["label"][i])
print(roc_auc_score(testY, pred))



# BERT TRAINING + CLASSIFICATION]
dataset = pd.read_excel("data/IP_dataset_reduced.xlsx")

numFalse = 0
numTrue = 0
counter = 0

order = random.sample(range(len(dataset)), 3000)
used = []
patient_id = []

for i in order:
    if (dataset["label"][i] == 0 and numFalse < 40):
        used.append(i)
        numFalse += 1
        patient_id.append(dataset["person_id"][i])
    elif (dataset["label"][i] == 1 and numTrue < 84):
        numTrue += 1
        used.append(i)
        patient_id.append(dataset["person_id"][i])

x, y = [], []
biobert = BiobertEmbedding()

snippets_df = pd.read_csv("data/IP_snippets_dataset.csv")
for i in range(len(snippets_df)):
    if snippets_df['person_id'][i] in patient_id:
        try:
            x.append(np.array(biobert.sentence_vector(snippets_df['snippet'][i])))
            y.append(snippets_df['label'][i])
        except: ''''''

x = np.array(x)
y = np.array(y)

class_weight = []
for label in y:
    if label == 0: class_weight.append(len(y)/(len(y)-sum(y)))
    else: class_weight.append(len(y)/sum(y))

text_clf = XGBClassifier(max_depth = 6, n_estimators= 100, booster = "gbtree", reg_alpha = 0.2, sample_weight=class_weight)
text_clf.fit(x,y)


preds = []
selected_snip = []
for i in range(len(dataset)):
    label = 0
    selected = ""
    if i in used:
        preds.append(np.NaN)
        selected_snip.append(selected)
    else:
        sents = sent_tokenize(dataset['snippets'][i])
        for sent in sents:
            try:
                prob = text_clf.predict_proba(biobert.sentence_vector(sent))
                if (prob[0][1] > label):
                    selected = sent
                    label = prob[0][1]
            except: ''''''
        preds.append(label)
        selected_snip.append(selected)

dataset["selected_snippet"] = selected_snip
dataset["CLASSIFICATION"] = preds
dataset.to_excel("data/IP_dataset_BERTsnippet_classified.xlsx")
