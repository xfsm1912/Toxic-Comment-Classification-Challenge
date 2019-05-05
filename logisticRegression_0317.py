import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from scipy import sparse

from subprocess import check_output
print(check_output(["ls", "../data"]).decode("utf8"))

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

feat_to_concat = ['comment_text']

allData = pd.concat([train[feat_to_concat],test[feat_to_concat]], axis=0)
allData.comment_text.fillna('unkown', inplace = True)

# Regex to remove all Non-Alpha Numeric and space
special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)

# regex to replace all numerics
replace_numbers = re.compile(r'\d+', re.IGNORECASE)

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Remove Special Characters
    text = special_character_removal.sub('', text)

    # Replace Numbers
    text = replace_numbers.sub('n', text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)

comments = []
for text in allData:
    comments.append(text_to_wordlist(text))

# test_comments = []
# for text in list_sentences_test:
#     test_comments.append(text_to_wordlist(text))

vect_word = TfidfVectorizer(max_features=50000, analyzer='word', ngram_range=(1,1))
vect_chars = TfidfVectorizer(max_features=20000, analyzer='char', ngram_range=(1,3))

all_words = vect_word.fit_transform(comments)
all_chars = vect_chars.fit_transform(comments)

train_new = train
test_new = test

train_words = all_words[:len(train_new)]
test_words = all_words[len(train_new):]

train_chars = all_chars[:len(train_new)]
test_chars = all_chars[len(train_new):]

train_feats = sparse.hstack([train_words, train_chars])
test_feats = sparse.hstack([test_words, test_chars])

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

only_col = ['toxic']

preds = np.zeros((test_new.shape[0], len(col)))

for i, j in enumerate(col):
    print('===Fit ' + j)

    model = LogisticRegression(C=4.0, solver='sag')
    print('Fitting model')
    model.fit(train_feats, train_new[j])

    print('Predicting on test')
    preds[:, i] = model.predict_proba(test_feats)[:, 1]

subm = pd.read_csv('../data/sample_submission.csv')

submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('feat_lr_cols_0317.csv', index=False)
