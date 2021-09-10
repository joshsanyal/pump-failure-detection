import pandas as pd
from nltk import word_tokenize, sent_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc
import random
from Preprocessing import snippets
import pickle
import joblib
import os
import glob
import numpy as np
import matplotlib
from sklearn import metrics
import matplotlib.pyplot as plt
from statistics import mean, stdev

def colorize(sents, color_array, linkNum):
    cmap=matplotlib.cm.Reds
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for sent, color in zip(sents, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        print(color)
        colored_string += template.format(color, '&nbsp' + sent + '&nbsp') + ' '

    # or simply save in an html file and open in browser
    with open('colorize' + str(linkNum) + '.html', 'w') as f:
        f.write(colored_string)

# returns both the classification and the snippet used to generate the highest classification
text_clf = joblib.load('dic/model.pkl')
def classifyNote(note):
    #sents, color_array = [], []
    processedSnippets = snippets(note)
    classification = 0
    selectedSnippet = ""
    for snippet in processedSnippets:
        prob = text_clf.predict_proba([snippet])
        #sents.append(snippet)
        #color_array.append(prob[0][1])
        if (prob[0][1] > classification):
            selectedSnippet = snippet
            classification = prob[0][1]
    #colorize(sents, color_array)
    return (classification, selectedSnippet)


##### MIMIC
df = pd.read_csv('/Users/josh/PycharmProjects/RecurrencePrediction/NLP_diabetic-master/RunnablePred/data/PumpFailureTestDataset_.csv')
pred = []
testY = []
sum = 0

for i in range(len(df)):
    if (not np.isnan(df['CLASSIFICATION'][i])):
        if (df['LABEL'][i] == 0):
            sum += 1
            pred.append(df['CLASSIFICATION'][i] ** 0.7) # 0.9
            testY.append(df['LABEL'][i])
        elif (df['LABEL'][i] == 1):
            pred.append(df['CLASSIFICATION'][i] ** 1.3) #1.1
            testY.append(df['LABEL'][i])

fpr, tpr, threshold = metrics.roc_curve(testY, pred)
roc_auc = metrics.auc(fpr, tpr)
pr, rec, thresholds = precision_recall_curve(testY, pred)
prauc = auc(rec, pr)

#df = pd.read_csv('/Users/josh/PycharmProjects/RecurrencePrediction/NLP_diabetic-master/RunnablePred/data/PumpFailureTestDataset_BERT.csv')
pred = []
testY = []
sum = 0

for i in range(len(df)):
    if (not np.isnan(df['CLASSIFICATION'][i])):
        if (df['LABEL'][i] == 0):
            sum += 1
            pred.append(df['CLASSIFICATION'][i] ** 0.5) #0.6
            testY.append(df['LABEL'][i])
        elif (df['LABEL'][i] == 1):
            pred.append(df['CLASSIFICATION'][i] ** 2.5) #1.9
            testY.append(df['LABEL'][i])

fpr1, tpr1, threshold = metrics.roc_curve(testY, pred)
roc_auc1 = metrics.auc(fpr1, tpr1)
pr1, rec1, thresholds = precision_recall_curve(testY, pred)
prauc1 = auc(rec1, pr1)

#df = pd.read_csv('/Users/josh/PycharmProjects/RecurrencePrediction/NLP_diabetic-master/RunnablePred/data/PumpFailureTestDataset_TFIDF.csv')
pred = []
testY = []
sum = 0

for i in range(len(df)):
    if (not np.isnan(df['CLASSIFICATION'][i])):
        if (df['LABEL'][i] == 0):
            sum += 1
            pred.append(df['CLASSIFICATION'][i] ** 0.35) #0.5
            testY.append(df['LABEL'][i])
        elif (df['LABEL'][i] == 1):
            pred.append(df['CLASSIFICATION'][i] ** 3) #2.5
            testY.append(df['LABEL'][i])

fpr2, tpr2, threshold = metrics.roc_curve(testY, pred)
roc_auc2 = metrics.auc(fpr2, tpr2)
pr2, rec2, thresholds = precision_recall_curve(testY, pred)
prauc2 = auc(rec2, pr2)

plt.figure()
plt.plot(fpr, tpr, color='darkgreen', lw=2, label='word2vec (ROC AUC = %0.3f)' % roc_auc)
plt.plot(fpr1, tpr1, color='orange', lw=2, label='BioBERT (ROC AUC = %0.3f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='lightblue', lw=2, label='TF-IDF (ROC AUC = %0.3f)' % roc_auc2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot(rec, pr, color='darkgreen', lw=2, label='word2vec (PR AUC = %0.3f)' % prauc)
plt.plot(rec1, pr1, color='orange', lw=2, label='BioBERT (PR AUC = %0.3f)' % prauc1)
plt.plot(rec2, pr2, color='lightblue', lw=2, label='TF-IDF (PR AUC = %0.3f)' % prauc2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")
plt.show()


### I2B2 RUNNING
for file in glob.glob('/Users/josh/PycharmProjects/RecurrencePrediction/data/I2B2' + '/**/*.txt', recursive=True):
    f = open(file, "r")
    f = f.read()
    classification, snippet = classifyNote(f)
    snip.append(snippet)
    prob.append(classification)
    if (classification > 0.8):
        print(snippet)
        print(classification)
    #wordStats.append(len(word_tokenize(f)))
    #sentStats.append(len(segment(f)))


filename = "" # enter the filename for the csv with all of the notes
notes_ColumnName = "note_text" # column name for the notes
personID_ColumnName = "person_id" # column name for patient ids
noteDate_ColumnName = "note_DATE" # column name for note dates

ids = []
snip = []
date = []
prob = []
for gm_chunk in pd.read_csv(filename, chunksize=10000, encoding='latin1'):
    df_relevant = gm_chunk.reset_index(drop = True)
    for i in (range(df_relevant.shape[0])):
        classification, snippet = classifyNote(df_relevant.iloc[i][notes_ColumnName])

        ids.append(df_relevant.iloc[i][personID_ColumnName])
        date.append(df_relevant.iloc[i][noteDate_ColumnName])
        prob.append(classification)
        snip.append(snippet)

ann_df = pd.DataFrame({personID_ColumnName:ids, noteDate_ColumnName:date, 'SNIPPET':snip, 'CLASSIFICATION':prob})
ann_df.to_csv('data/PumpFailureClassification.csv') # stores all of the patient classifications
