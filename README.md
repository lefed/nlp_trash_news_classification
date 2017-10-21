# nlp_trash_news_classification
python nlp project and Flask app to classify online articles' likeness to news, tabloids, and hyper-partisan trash

### Project purpose
I delved into this project in order to see if there were differentiating factors between the 'real' news, tabloids, and hyper-partisan trash found in online articles. I wanted to see if symantic differences could pick up how similar content was to each of these sources, and if trashy, non-fact based news was more akin to tabloid articles than verified news sources.

### Methodology
In order to gather content for article classification, I used Selenium and BeautifulSoup to scrape article content from:

* News Articles - _I gathered 400 unique articles from [CBC.ca's News Section](http://www.cbc.ca/news) to train my news classifier._
* Tabloid Articles - _I gathered over 200 unique articles from [Star Magazine's News Section](http://starmagazine.com/category/news/) to train my tabloid classifier._
* Hyperpartisan Trash - _I gathered over 200 unique articles from the Breitbart North of the web, [Rebel Media](https://www.therebel.media/) to train my trash classifier._

After parsing the content of these articles and storing just the text associated with the headlines and body content, I proceeded with my nlp work.

In order to evaluate article headlines, I tokenized and count vectorized the texts, removed custom stop words that would be too related to topic - such as removing political terms, all common first names, celebrity names, and words that seems to be specific to the sources of the content.

I also created a custom punctuation list for punctuation weighting, tagged the parts of speech of the headlines, looked at length of sentences and number of sentences, and also the subjectivity and sentiment score of the text.

In order to evaluate article body text, I count vectorized the content with custom stop words and punctuation, then used part of speech tagging and removed all subject words leaving only the parts of speech patterns of the article content - ensuring that I would be able to look at the semantics of the language instead of picking up on specific topics and classifying articles based on counts of things such as political content. 

### Results

My classifiers were able to achieve excellent accuracy after my text cleaning and weighting of factors mentioned above. Even after removing the actual topic 'content' by only using the parts of speech of the article bodies, my classifier was able to maintain an accuracy of over 95%.

### What was important for content scoring?

Things that showed up as extremely important to my classification included:

* Punctuation use: _News articles tend to use more notations of quotes and sources -. Tabloids tend to use exclamation points!. Opinion and rhetoric tends to use question marks?._

* Repetition of part of speech phrasing: _News articles tend to use possesive endings and 3rd person verbs as they describe events and quote sources. Tabloids tend to use many prepositions and discuss people and their actions in third person present form. Opinion and rhetoric tends to use determiners (noun modifications) followed by proper nouns i.e. use language that expresses a classification or pre-determined judgement of a person, group, or organization._

* Language patterns: _News articles tend to work through descriptions and references. Opinion and rhetoric tends to repeat questioning, judgements, and descriptions of actions. Tabloids tend to repeat descriptions of people, common nouns, and phrases like 'in her/his' or 'a source says'._

### What was important for headline scoring?

* Punctuation use: _Tabloids and opinion/rhetoric tends to use ! and ?_

* Length of headline and number of sentences: _News articles tend to have longer, single sentence headlines. Tabloids tend to have short, mulit-sentence headlines._

* Sentiment score: _News articles tend to be neutral, tabloids negative, and rhetoric positive in sentiment scoring._

* Subjectivity score: _News articles tend to be objective, opinion tends to be subjective._

* Use of specific words and phrasings: _Gossip and rhetoric have key words and phrasings._

* Language patterns: _Language style tends to vary between publications and headline types._

## How to run the Flask App
If you would like to try out the app, download the files from this repo and run form_app2.py from command line. Navigate your browser to the address where the app is running. You should see the following format:

![](https://github.com/lefed/nlp_trash_news_classification/blob/master/images/flask_app_landing.png)

Choose an online article to evaluate - for example we could use the following [CBC News Article](http://www.cbc.ca/news/world/spain-cabinet-catalonia-1.4365757)

This particular article returns as being predominantly like news - both for its headline and content as can be seen from the screenshots below.

![](https://github.com/lefed/nlp_trash_news_classification/blob/master/images/example_CBC_article_1.png)

![](https://github.com/lefed/nlp_trash_news_classification/blob/master/images/example_CBC_article_2.png)

![](https://github.com/lefed/nlp_trash_news_classification/blob/master/images/example_CBC_article_3.png)

![](https://github.com/lefed/nlp_trash_news_classification/blob/master/images/example_CBC_article_4.png)

