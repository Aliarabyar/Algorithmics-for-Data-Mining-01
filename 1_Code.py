# Algorithmics for Data Mining
# Deliverable 1: Cyberbullying Detection on Social Media
# Ali Arabyarmohammadi
# March 2022


# Load Libraries

import pandas as pd
import numpy as np
import re
import nltk

from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

np.random.seed(100)


#


df = pd.read_csv('1_kaggle_parsed_dataset.csv', sep="," , engine = 'c' ,encoding='latin1')
df.head(10)


#


#df = pd.concat([df[df['code']==0].sample(n=1000),df[df['code']==1].sample(n=1000)])
Corpus = df[['text' , 'code']]
Corpus=  Corpus.rename({'code': 'label' , 'text':'text'} ,  axis=1)
Corpus = Corpus.reset_index(drop=True)
print(Corpus.head (10))
cat = Corpus.groupby("label").count()
print(cat)


#


start_time = datetime.now()
print('Start: {}'.format(start_time))

# Step - a : Remove blank rows if any.
Corpus['text'].dropna(inplace=True)
Corpus['label'].dropna(inplace=True)
Corpus['text'] = [entry.lower() for entry in Corpus['text']]
Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]

for index,entry in enumerate(Corpus['text']):
    Final_words = []
    for word in entry:
        if word not in stopwords.words('english'):
            word = re.sub("\d+$", " ", word)
            word =  re.sub(r"\b[a-zA-Z]\b", "", word) # Reomve single Chars: "this is a test" -> 'this is  test'
            word = re.sub('[^ A-Za-z0-9]+', '', word) # Remove Special Chars: "this is @ test" -> "this is  test"
            word = re.sub(r',', ' ', word) # Remove ','
            word = re.sub(' +', ' ', word) # Remove multiple spaces: 'this     is   a  test' -> 'this is a test'
            
            Final_words.append(word)

    Corpus.loc [index,'text'] = str(Final_words)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


#


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text'],Corpus['label'],test_size=0.2)


#


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y) 


#


Tfidf_vect = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
Tfidf_vect.fit(Corpus['text'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


#


from sklearn.linear_model import LogisticRegression
start_time = datetime.now()
print('Start: {}'.format(start_time))
# fit the training dataset on the classifier
lr = LogisticRegression()
lr.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_lr = lr.predict(Test_X_Tfidf)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
#Use accuracy_score function
print("LogisticRegression Accuracy Score -> ",accuracy_score(predictions_lr, Test_Y)*100)
print("LogisticRegression Recall Score -> ",recall_score(predictions_lr, Test_Y )*100)
print("LogisticRegression Precision Score -> ",precision_score(predictions_lr, Test_Y)*100)
print("LogisticRegression F1 Score -> ", f1_score(predictions_lr, Test_Y )*100)
print("LogisticRegression", ",", end_time - start_time, ",",accuracy_score(predictions_lr, Test_Y)*100, ",", recall_score(predictions_lr, Test_Y )*100, ",", precision_score(predictions_lr, Test_Y)*100, ",", f1_score(predictions_lr, Test_Y )*100)


#


from sklearn.ensemble import RandomForestClassifier
start_time = datetime.now()
print('Start: {}'.format(start_time))
# fit the training dataset on the classifier
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_rf = clf.predict(Test_X_Tfidf)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
#Use accuracy_score function
print("RandomForest Accuracy Score -> ",accuracy_score(predictions_rf, Test_Y)*100)
print("RandomForest Recall Score -> ",recall_score(predictions_rf, Test_Y )*100)
print("RandomForest Precision Score -> ",precision_score(predictions_rf, Test_Y)*100)
print("RandomForest F1 Score -> ", f1_score(predictions_rf, Test_Y )*100)
print("RandomForest", ",", end_time - start_time, ",",accuracy_score(predictions_rf, Test_Y)*100, ",", recall_score(predictions_rf, Test_Y )*100, ",", precision_score(predictions_rf, Test_Y)*100, ",", f1_score(predictions_rf, Test_Y )*100)


#


start_time = datetime.now()
print('Start: {}'.format(start_time))
# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y )*100)
print("Naive Bayes Recall Score -> ", recall_score(predictions_NB, Test_Y)*100)
print("Naive Bayes Precision :", precision_score(predictions_NB, Test_Y)*100)
print("Naive Bayes F1 Score :", f1_score(predictions_NB, Test_Y)*100)
print("Naive Bayes", ",",end_time - start_time, ",",accuracy_score(predictions_NB, Test_Y)*100, ",", recall_score(predictions_NB, Test_Y )*100, ",", precision_score(predictions_NB, Test_Y)*100, ",", f1_score(predictions_NB, Test_Y )*100)


#


start_time = datetime.now()
print('Start: {}'.format(start_time))
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print("SVM Recall Score -> ",recall_score(predictions_SVM, Test_Y)*100)
print("SVM Precision Score -> ",precision_score(predictions_SVM, Test_Y)*100)
print("SVM F1 Score -> ", f1_score(predictions_SVM, Test_Y)*100)
print("SVM", ",", end_time - start_time, ",",accuracy_score(predictions_SVM, Test_Y)*100, ",", recall_score(predictions_SVM, Test_Y )*100, ",", precision_score(predictions_SVM, Test_Y)*100, ",", f1_score(predictions_SVM, Test_Y )*100)




