import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





data=pd.read_csv("D:\\DATA_SCIENCE\\readmycourse\\study_materials\\NLP\\fakenews.csv")

data.isnull().sum()

message=data[["title","label"]]

message.isnull().sum()
message=message.dropna()

message=message.reset_index()
message=message.drop(["index"],axis=1)

ls=WordNetLemmatizer()


corpus=[]
for i in range(0,len(message["title"])):
    review=re.sub("[^a-zA-Z]"," ",message["title"][i])
    review=review.lower()
    review=review.split()
    review=[ls.lemmatize(j) for j in review if j not in set(stopwords.words("english"))]
    review=" ".join(review)
    corpus.append(review)
    

# ------------------------------------------Applying Countvectorizer & TfidfVectorizer-----------------------------------
#----- Creating the Bag of Words model----------------------------------------------------------------
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
X = tf.fit_transform(corpus).toarray()


# Divide the dataset into Train and Test
y=message["label"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)


count_df = pd.DataFrame(X_train, columns=tf.get_feature_names())



#------------------------------------------------classifier----------------------------------------
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)


#----------------------------------------classification report-------------------------------------
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(pred,y_test))


#--------------------------------PASSIVE AGRESSIVE CLASSIFIER ALGORITHM----------------------------
from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier()
linear_clf.fit(X_train,y_train)
y_pred=linear_clf.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
from sklearn import metrics
#--------------------------------------Multinomial Classifier with Hyperparameter-------------------

previous_score=0.90
for i in np.arange(0,1.1,.1):
    mchp=MultinomialNB(alpha=i)
    mchp.fit(X_train,y_train)
    y_pred=mchp.predict(X_test)
    new_score=metrics.accuracy_score(y_test,y_pred)
    print("accuracy : %0.3f" %new_score,"alpha:%0.3f"%i)































