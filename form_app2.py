#this file represents the nlp classifier flask app

from flask import request, redirect, render_template
import Python_Helper_Utils
from Python_Helper_Utils import make_string_class, make_df_class, pos_tag_and_flat_class, replace_names_class, TextStats, pos_tag_only_class
from Python_Helper_Utils import DenseTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import flask
from sklearn.externals import joblib
from flask import Flask
from flask import request
from flask import render_template

#----------MODEL IN MEMORY---------------#

#load model created for headline classification
headline_classification_model = joblib.load('news_headline_classification_model.pkl')

#load model created for content classification
news_content_classification_model = joblib.load('news_content_classification_model.pkl')

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage

@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, index2.html
    """
    return render_template('index2.html', headline=None, news_likeness=None, tabloid_likeness=None, partisan_opinion_likeness=None, content=None, content_news_likeness=None, content_partisan_opinion_likeness=None, content_tabloid_likeness=None )


@app.route('/', methods = ['POST', 'GET'])
def get_eval_content():
    print('CONTENT')

    headline = None
    content = None
    news_likeness = None
    partisan_opinion_likeness = None
    tabloid_likeness = None
    cont_news_likeness = None
    cont_partisan_opinion_likeness = None
    cont_tabloid_likeness = None
    
#    if request.method == 'POST':
    print(request.form)

    if request.form['content']:
        content = request.form['content']

        if content != None:
            parsed_content = []
            parsed_content.append(''.join(content))
            print("The user input content of:'" + content[:20] + "'")

            cont_pred_proba = news_content_classification_model.predict_proba(parsed_content)
            cont_pred_proba_content = np.split(cont_pred_proba, 3, axis =1)
            cont_news_likeness = str(cont_pred_proba_content[0]*100).replace('[', '').replace(']', '')
            cont_news_likeness = cont_news_likeness[:4]
            cont_partisan_opinion_likeness = str(cont_pred_proba_content[1]*100).replace('[', '').replace(']', '')
            cont_partisan_opinion_likeness = cont_partisan_opinion_likeness[:4]
            cont_tabloid_likeness = str(cont_pred_proba_content[2]*100).replace('[', '').replace(']', '')
            cont_tabloid_likeness = cont_tabloid_likeness[:4]
            print("news%:", cont_news_likeness[:4])
            print("trash%:", cont_partisan_opinion_likeness[:4])
            print("tabloid%:", cont_tabloid_likeness[:4])

    if request.form['headline']:
        headline = request.form['headline']
            
        if headline != None:
            parsed_headline = []
            parsed_headline.append(''.join(headline))
            print("The user input a headline of:'" + headline + "'")

            y_pred_proba = headline_classification_model.predict_proba(parsed_headline)
            y_pred_proba_headline = np.split(y_pred_proba, 3, axis =1)
            news_likeness = str(y_pred_proba_headline[0]*100).replace('[', '').replace(']', '')
            news_likeness = news_likeness[:4]
            partisan_opinion_likeness = str(y_pred_proba_headline[1]*100).replace('[', '').replace(']', '')
            partisan_opinion_likeness = partisan_opinion_likeness[:4]
            tabloid_likeness = str(y_pred_proba_headline[2]*100).replace('[', '').replace(']', '')
            tabloid_likeness = tabloid_likeness[:4]
            print("news%:", news_likeness[:4])
            print("trash%:", partisan_opinion_likeness[:4])
            print("tabloid%:", tabloid_likeness[:4])

    return render_template('index2.html', content=content, cont_news_likeness=cont_news_likeness, cont_partisan_opinion_likeness=cont_partisan_opinion_likeness, cont_tabloid_likeness=cont_tabloid_likeness, headline = headline, news_likeness=news_likeness, partisan_opinion_likeness=partisan_opinion_likeness, tabloid_likeness=tabloid_likeness)
            
            

    

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0')
app.run(debug=True)
