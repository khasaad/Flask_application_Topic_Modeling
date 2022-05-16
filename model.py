import nltk
import re
import spacy

import logging
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from spacy.lang.en.stop_words import STOP_WORDS as en_stop

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# Create and configure logger
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename="test.log",
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d-%m-%y %H:%M %S',
                    filemode='w')

logger = logging.getLogger()

# Script execution
logging.warning("Script Started : " + datetime.today().strftime("%d/%m/%Y %H %M %S"))
# Test the logger
logger.info("Read data")

# Read data
data_lemmatized = pd.read_csv('C:/Users/khale/Topic modeling/data_npr_cleaning.csv', parse_dates=[0],
                              infer_datetime_format=True)

logger.info("Run cleaned function")


def clean_txt(txt: str) -> str:
    txt = re.sub(r'[^\w]', ' ', txt)
    return txt


def remove_digit(txt: str) -> str:
    result = ''.join([i for i in txt if not i.isdigit()])
    return result


list__stopWords = list(en_stop)


def remove_stop_words(txt: str):
    string_txt = " "
    list_words = []

    sentence = txt.split()

    for val in sentence:
        if val not in list__stopWords:
            list_words.append(val)

    return string_txt.join(list_words)


lemmatizer = WordNetLemmatizer()


def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatization(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))

    return " ".join(res_words)


# English stop words

list_stopWords = ['year', 'time', 'know', 'come', 'use', 'tell', 'want', 'day', 'say', 'says', 'ha', 'wa', 'like',
                  'think', 'make', 'new', 'song', 'time', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                  'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                  'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they',
                  'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                  'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                  'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                  'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                  'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                  'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                  'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                  'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
                  'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
                  "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                  "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
                  "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
                  "wouldn't", 'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                  'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's',
                  'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z']


def stop_words(txt: str):
    string_txt = " "
    list_words = []

    sentence = txt.split()

    for val in sentence:
        if val not in list_stopWords:
            list_words.append(val)

    return string_txt.join(list_words)


logger.info("Start training model")

data_lemmatized = data_lemmatized.loc[:, ~data_lemmatized.columns.str.contains('^Unnamed')]
data_lemmatized['Article'] = data_lemmatized['Article'].apply(lambda x: stop_words(x))

# BOW
vectorizer = CountVectorizer(analyzer='word', min_df=10, stop_words='english',
                             lowercase=True, token_pattern='[a-zA-Z0-9]{3,}',
                             max_features=50000, )

data_vectorized = vectorizer.fit_transform(data_lemmatized['Article'])

# Load the model
model = joblib.load("C:/Users/khale/Topic modeling/best_model_lda_BOW_GSCV.pkl")

# Best Model
best_lda_model = model.best_estimator_

# lda_output_GSCV = best_lda_model.transform(data_vectorized)
lda_output_GS = best_lda_model.transform(data_vectorized)


# Topic-Keyword Matrix
def show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


logger.info("End training model")

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)
# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)

nlp = spacy.blank("en")


def predict_topic(text, nlp=nlp):
    # Step 1: Clean with simple_preprocess
    mytext_1 = clean_txt(text[0])

    # Step 2: Remove digit
    mytext_2 = remove_digit(mytext_1)

    # Step 3: Remove specific words
    mytext_3 = stop_words(mytext_2)

    # Step 4: Remove stop words
    mytext_4 = remove_stop_words(mytext_3)

    # Step 5: Lemmatize
    mytext_5 = lemmatization(mytext_4)

    # Step 6: Vectorize transform
    mytext_6 = vectorizer.transform([mytext_5])

    # Step 7: LDA Transform
    topic_probability_scores = best_lda_model.transform(mytext_6)
    topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), 1:14].values.tolist()

    return mytext_5, topic, topic_probability_scores


logging.warning("Script Ended : " + datetime.today().strftime("%d/%m/%Y %H %M %S"))
