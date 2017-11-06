import pandas as pd
import numpy as np
from sklearn import datasets, random_projection 
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin, BaseEstimator
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize

import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score,roc_curve,auc

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()


import re

#removal of extra punctuation from scrapped text
remove_bracket_content = lambda x: re.sub("[\(\[].*?[\)\]]", "", str(x))

#split header text based on new line breaks
header_split = lambda x: str(x).split('\n')

#helper function to parse header content
def header_parse(x):
    
    if 'LIVE' in x:
        title = ''.join(str(i)+' ' for i in x)
        topic = 'News Livestream'
        author = 'none'

    elif 'LIVE' not in x and len(x) >0:
        title = ''.join(str(i)+' ' for i in x[:-2])
        topic = str(x[0])
        author = str(x[-1])
        
    return title, topic, author

#helper function to parse title content
def title_parse(x):
        
    if 'LIVE' in x:
        title = ''.join(str(i)+' ' for i in x)

    elif 'LIVE' not in x and len(x) >0:
        title = ''.join(str(i)+' ' for i in x[:-2])

    return title

#helper function to parse topic from text
def topic_parse(x):
    
    if 'LIVE' in x:
        topic = 'News Livestream'

    elif 'LIVE' not in x and len(x) >0:
        topic = str(x[0])
        
    return topic


#helper function to parse author from text
def author_parse(x):
    
    if 'LIVE' in x:
        author = 'none'

    elif 'LIVE' not in x and len(x) >0:
        author = str(x[-1])
        
    return author


#helper function to parse news topic
def news_topic_parse(x):
    
    url_str = str(x)
    
    if re.search('news/(.*)/',url_str):
        news_topic = re.search('news/(.*)/',url_str).group(1)
        return news_topic
    
    else:
        return 0
    


#helper function to parse headlines from rebel
def rebel_headline_parse(x):
    if len(x)>1:
        headline = str(x[1])
    else:
        headline = 0
    return headline

#helper function to parse date 
def rebel_publish_date_parse(x):
    if len(x)>0:
        publish_date = str(x[0])
    else:
        publish_date = 0
    return publish_date

#helper function to parse author from rebel text
def rebel_author_parse(x):
    if len(x)>2:
        author = str(x[2])
    else:
        author = 0
    return author


#helper function for tabloid date and author
def star_get_author_date(x):
    if len(x)>1:
        author_date = str(x[-1])
    else:
        author_date = 0
    return author_date

#helper function to parse star headlines
def star_get_headline(x):
    if len(x)>0:
        headline = ''.join(str(i)+' ' for i in x[:-1])
    else:
        headline = 0
    return headline    
    


#helper function to apply to text information to ensure proper string format without extra spaces
make_string = lambda x: str(x).strip()


def make_string(x):
    return(str(x).strip())


#helper function to remove extra punctuation and clean text
def clean_string(x):
    ''.join(x.split("\\"))
    x.replace('( )', '').replace('. "', '."')
    return x


#list of custom stop words to be used for nlp - these are trigger words or punctuation that should not be used to allow for source prediction
custom_stop_words =['warning', 'star', 'language warning', 'cbc', 'Rebel', 'language', '( )', '. “','. ``', '00', '( )', 'i' , 'a', 'magazine', 'ok', 't', '``', 'weekly', '“', '”','s', '‘', "'s", '’', "'re", "n't", 'didn', 'powered', "ikea®", 'www.ikea.com/us/kitchens', 'ikea', 've', 'aug.', 'ca', 'l', 'la', 'rebel', "'m", 'kitchen', 'kitchens', 'quantico' ]


#customized punctuation list so as to allow for understanding of how !, ?, and - are used but not use other punctuation counts
custom_punct = list(string.punctuation)
custom_punct.append('""')
custom_punct.append("''")
custom_punct.remove('!')
custom_punct.remove('?')
custom_punct.remove('-')


#count vectorizer
count_vect = CountVectorizer(analyzer='word', stop_words=custom_stop_words + custom_punct, tokenizer=word_tokenize)


#helper function to use part of speech tagging and include in full sentence of text
def pos_tag_and_flat(x):
    text = nltk.tokenize.word_tokenize(x)
    pos = nltk.pos_tag(text)
    x_flat = [e for l in pos for e in l]
    str1 = ' '.join(str(e) for e in x_flat)
    return str1


#helper function to use only part of speech tagging and custom punctuation to understand language syntax
def pos_tag_only(x):
    text = nltk.tokenize.word_tokenize(x)
    pos = nltk.pos_tag(text)
    x_pos = [tup[0] if '!' in tup else tup[0] if '?' in tup else tup[0] if '-' in tup else tup[1] for tup in pos ]
    x_pos_str = ' '.join(str(e) for e in x_pos)
    return x_pos_str


#helper function to replace names so as not to provide easy prediction of source
def replace_names(text):
    first_names_df = pd.read_csv('first_names.txt', header = None, sep = ',')
    most_common_first_names_df = first_names_df.nlargest(800, [2]).reset_index()
    common_names = ['Brad', 'Harry', 'Kris', 'Kylie', 'Don', 'Madonna', 'Shia', 'Tamron', 'Kim', 'Gwenyth', 'Leonardo', 'Mathew', 'Macaulay', 'Farrah', 'Beckinsale', 'Dale', 'Polanski', 'A-Rod', 'Bette', 'Mel', 'Bella']
    for n in range(0, len(most_common_first_names_df)):
        common_names.append(most_common_first_names_df[0][n])
    for name in common_names:
        new_text = text.replace(name, 'NAME1')
    return (str(new_text))


#sentence tokenizer
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


#helper class to transform content to Dense Matrix to allow for proper predictive model input 
class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


#helper class to convert information into required strings from multiple documents
class make_string_class(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, docs):
        string_class = [make_string(text)
                for text in docs]
        string_series = pd.Series(string_class)
#        print("after make_string_class the type is:", type(string_series), string_series.shape)
        return(string_series)


#helper class to do part of speech tagging and flattening
class pos_tag_and_flat_class(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, docs):
        pos_tag_list = [pos_tag_and_flat(text) for text in docs.iloc[:, 0]]
        pos_tag_series = pd.Series(pos_tag_list)
#        print("after pos_tag_and_flat_class:", type(pos_tag_series), pos_tag_series.shape)
        return(pos_tag_series)


#helper class to do part of speech tagging and return only POS and punctuation
class pos_tag_only_class(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, docs):
        pos_tag_list = [pos_tag_only(text) for text in docs.iloc[:, 0]]
        pos_tag_series = pd.Series(pos_tag_list)
#        print("after pos_tag_and_flat_class:", type(pos_tag_series), pos_tag_series.shape)
        return(pos_tag_series)


#helper class to remove common names
class replace_names_class(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, docs):
        replace_names_data = [replace_names(text) for text in docs]
        name1_names = pd.Series(replace_names_data)
#        print("after replace_names_class:", name1_names.shape)
        return(name1_names)


#helper class for dataframe creation for formatting for model
class make_df_class(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, docs):
        list_df = []
        for text in docs:
            list_df.append(text)
        df_mid = pd.DataFrame(list_df, columns = ['input'])
        print("after make_df_class:", type(df_mid), df_mid.shape)
        return(df_mid)

#pipeline for cleaning and parsing text data
clean_content_pipeline = Pipeline([
    ('makestr', make_string_class()),
    ('make_df', make_df_class()),
    ('pos_tag_flat', pos_tag_only_class()),
    ('name1', replace_names_class())
])


#pipeline for cleaning, vectorizing, and modeling
ppl1 = Pipeline([
              ('clean_input', clean_content_pipeline),
              ('vectorizer', CountVectorizer(ngram_range=(1, 3), analyzer='word', stop_words=custom_stop_words + custom_punct, tokenizer=word_tokenize)),
              ('to_dense', DenseTransformer()),
              ('clf',   LogisticRegression())
      ])

# train the classifier AFTER applying tagging
#content_model = ppl1.fit(X, y)

# test the classifier AFTER applying tagging
#y_pred = model.predict(X_test)
#y_pred_proba = model.predict_proba(X_test)


from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob


#stop words list for headlines
headline_stop_words = ['Trudeau', 'Trump', 'alberta', 'live','bc', 'listen', 'trump', 'trudeau', 'mcinnes', 'name1m', 'daily', 'roundup', 'streams', 'stream', 'eclipse', 'toronto', 'edge', 'celebrity', 'montreal', 'solar', 'radio', 'introducing', 'notley', 'tommy', 'robinson', 'ann', 'coulter', 'thicke', 'kardashian', 'kim']


#helper class to create numerical statistics of text data to be weighted and used in model for prediction
class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self
    print("Here at TextStats")    
    def transform(self, docs):
        return [{'length': len(text),
                 'num_sentences': text.count('.'),
                'sentiment_score_pos': TextBlob(text).sentiment[0],
                'subjectivity_score': TextBlob(text).sentiment[1]}
                for text in docs]

#pipeline for text stat creation and vectorization
text_stats_pipeline = Pipeline([
                    ('stats', TextStats()),
                    ('vect', DictVectorizer())
                ])


#pipeline for cleaning and parsing headlines
clean_headline_pipeline = Pipeline([
    ('makestr', make_string_class()),
    ('make_df', make_df_class()),
    ('pos_tag_flat', pos_tag_and_flat_class()),
    ('name1', replace_names_class())
])


#pipeline for cleaning and predicting headline classification
ppl2 = Pipeline([
        ('clean_headline', clean_headline_pipeline),
        ('union', FeatureUnion(
            transformer_list=[
                ('textstats', text_stats_pipeline),

                ('headline_analysis', Pipeline([
                    ('vectorizer', CountVectorizer(ngram_range=(1, 3), analyzer='word', stop_words=custom_stop_words + custom_punct + headline_stop_words, tokenizer=word_tokenize)),
                ]))
           ],
            transformer_weights= {
            'textstats': 1.0,
            'headline_analysis': 1.0,
        },
        )),
              ('to_dense', DenseTransformer()),
              ('clf',   LogisticRegression())
      ])
         
    
# train the classifier AFTER applying tagging
#headline_model = ppl2.fit(X, y)

