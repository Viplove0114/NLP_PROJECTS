import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


df=pd.read_csv("SMSSpamCollection",sep="\t",header=None,names=["class","text"])
ls=WordNetLemmatizer()



corpus=[]
for i in range(len(df["text"])):
    review=re.sub("[^a-zA-Z]"," ",df["text"][i])
    review=review.lower()
    review=review.split()
    review=[ls.lemmatize(j) for j in review if j not in set(stopwords.words("english"))]
    review=" ".join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
X=tf.fit_transform(corpus).toarray()

y=pd.get_dummies(df["class"],drop_first=True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=0)

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()

spam_model=nb.fit(x_train,y_train)

y_pred=spam_model.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
con_m=confusion_matrix(y_test,y_pred)
print(classification_report(y_pred,y_test))

a=tf.get_feature_names()
a
col_names=pd.DataFrame(X,columns=a)
