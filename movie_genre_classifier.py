import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


data=pd.read_csv("D:\\DATA_SCIENCE\\readmycourse\\study_materials\\NLP\\movie_genre.csv")
data.duplicated().sum()
data.isnull().sum()
data['genre'].unique()
data.drop('id',axis=1,inplace=True)
message=data[["text","genre"]]


special_character_remover = re.compile('[/(){}\[\]\|@,;]')
extra_symbol_remover = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()
    text = special_character_remover.sub(' ', text)
    text = extra_symbol_remover.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text
message['text']=message['text'].apply(clean_text)

X = message.text
y = message.genre
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)


#LOGISTIC REGRESSION
lr = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', LogisticRegression(penalty='none')),
              ])

lr.fit(X_train,y_train)
y_pred1 = lr.predict(X_test)
print(f"Accuracy is : {accuracy_score(y_pred1,y_test)}")



#NAIVE BAYES
naivebayes = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
naivebayes.fit(X_train, y_train)

y_pred = naivebayes.predict(X_test)
print(f'accuracy {accuracy_score(y_pred,y_test)}')


#XG BOOSTING
xgboost = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', XGBClassifier()),
              ])
xgboost.fit(X_train, y_train)

y_pred = xgboost.predict(X_test)
print(f'accuracy {accuracy_score(y_pred,y_test)}')


#RANDOM FOREST
rf= Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', RandomForestClassifier(n_estimators=76, max_features='auto'))])
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print( f'accuracy is: {accuracy_score(y_test, y_pred)}')


# LOGISTIC REGRESSION GIVES BEST ACCURACY i.e = 93%






