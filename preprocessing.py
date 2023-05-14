import pandas as pd
import numpy as np
import re
import string

train = pd.read_csv('/kaggle/Datasets/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('./kaggle/Datasets/tweet-sentiment-extraction/test.csv')

train.dropna
test.dropna

def jaccard(s1, s2):
    a = set(s1.lower().split())
    b = set(s2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) + len(c))

results_jaccard = []

for ind, row in train.iterrows():
    sentence1 = row.text
    sentence2 = row.selected_text

    jaccard_score = jaccard(sentence1, sentence2)
    results_jaccard.append([sentence1, sentence2, jaccard_score])

jaccard = pd.Dataframe(results_jaccard, columns=["text", "selected_text", "jaccard_score"])
train.merge(jaccard, how = 'outer')

train['Num_words_ST'] = train['selected_text'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text
train['Num_word_text'] = train['text'].apply(lambda x:len(str(x).split())) #Number Of words in main text
train['difference_in_words'] = train['Num_word_text'] - train['Num_words_ST'] #Difference in Number of words text and Selected Text

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

train['text'] = train['text'].apply(lambda x:clean_text(x))
train['selected_text'] = train['selected_text'].apply(lambda x:clean_text(x))

test['text'] = test['text'].apply(lambda x:clean_text(x))
test['selected_text'] = test['selected_text'].apply(lambda x:clean_text(x))

train.head() # cleaned train set

test.head() # cleaned test set