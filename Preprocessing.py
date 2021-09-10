import nltk
import re
import string
from nltk.stem import WordNetLemmatizer
from dateutil.relativedelta import relativedelta
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


def snippets(rawnote):
    keywords = pd.read_csv("dic/Dictionary.csv")
    keywords = keywords["Terms"]

    rawnote = rawnote.split('PLAN:')[0]
    str1 = rawnote
    str1 = ' '+str1+' '
    snippets = []
    str1 = re.sub(r'[^\x00-\x7f]',r' ', str1)
    re1='(\\d+)'
    re2='(\\.)'
    re3='( )'
    re4='((?:[a-z][a-z]+))'
    rg = re.compile(re1+re2+re3+re4,re.IGNORECASE|re.DOTALL)
    str1 = re.sub(rg,r'\n', str1)
    str1 = re.sub('\s\s\s+','\n',str1)
    str1 = str1.replace('\r', '\n')
    str1 = str1.replace('#', '\n')
    paragraphs = [p for p in str1.split('\n') if p]
    for paragraph in paragraphs:
        temp_sentences = sent_tokenize(paragraph) #replace?
        for i in range(len(temp_sentences)):
            temp_sentences[i] = temp_sentences[i].lower()
            if (checkBlockedWord(temp_sentences[i])):
                for term in keywords:
                    if re.search(r'\b' + term.lower() + r'\b', temp_sentences[i]):
                        snippets.append(elimExcludeWords(temp_sentences[i]))
    return list(dict.fromkeys(snippets))


def checkBlockedWord(snippet):
    blocked_words = ["directed", "back up", "backup", "company", "discuss", "verbalize", "scenario", "troubleshoot", "review", "event", "case", "plan","no pump failure", "no pump malfunction", "no failure", "no malfunction"]
    for term in blocked_words:
        if re.search(r'\b' + term.lower() + r'\b', snippet):
            return False
    return True

def elimExcludeWords(snippet):
    exclude_list = ['hour', 'aspect', 'downward', 'could', 'including', 'represents', 'follow', 'described', 'noted', 'pm', 'given', 'representing', 'along', 'well', 'causing', 'versus', 'more', 'could', 'involving', 'third', 'fourth', 'size', 'along', 'mm', 'also', 'right', 'left', 'seen', 'measuring', 'cm', 'amount', 'age', 'approximately', 'clinical', 'showing', 'year', 'old', 'finding', 'impression', 'may', 'represent', 'appears', 'identify', 'identified', 'identified', 'comment', 'additional', 'though', 'narrative', 'relative', 'comparison', 'history', 'compared', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "youre", "youve", "youll", "youd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "shes", 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "thatll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'both', 'each', 'other', 'such', 'only', 'own', 'so', 'than', 's', 't', 'can', 'will', 'just', 'don', "dont", 'should', "shouldve", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "arent", 'couldn', "couldnt", 'didn', "didnt", 'doesn', "doesnt", 'hadn', "hadn't", 'hasn', "hasnt", 'haven', "havent", 'isn', "isnt", 'ma', 'mightn', "mightnt", 'mustn', "mustnt", 'needn', "neednt", 'shan', "shant", 'shouldn', "shouldnt", 'wasn', "wasnt", 'weren', "werent", 'won', "wont", 'wouldn', "wouldnt"]
    resultwords = [word for word in word_tokenize(snippet) if word not in exclude_list]
    return process(' '.join(resultwords)) #remove punctuation


def prelimProcess(text):
    # STANDARDIZE WHITESPACES
    spaces = "                         "
    for i in range(22):
        text = text.replace(spaces[i:],"\n")
    newlines = "\n\n\n\n\n\n\n\n\n\n"
    for i in range(10):
        text = text.replace(newlines[i:],"\n")
    return text


def process(snippet):
    snippet = prelimProcess(snippet)
    snippet = snippet.replace("\n", " ")
    snippet = re.sub('['+ string.punctuation + ']', ' ', snippet) #remove punctuation
    snippet = snippet.lower()

    words = snippet.split(' ')
    sent = []
    for j in range(len(words)):
        if (len(words[j]) > 1 and not words[j].isnumeric()):
            sent.append(words[j])
    return " ".join(sent)
