# Project

 - - - - Pre-Processing - - - - 
 
install.packages("tm") # for text mining 
install.package("SnowballC") # for text stemming, download from net and add from local drive
install.packages("wordcloud") # word-cloud generator 
install.packages("RColorBrewer") # color palettes

library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")

setwd("C:/Users/Anam/Desktop/Full Experiment/RawDatasets/Security")
source<- DirSource("C:/Users/Anam/Desktop/Full Experiment/RawDatasets/Security")
YourCorpus<- Corpus(source, readerControl=list(reader=readPlain)) 
docs<- VCorpus(VectorSource(YourCorpus))
toSpace<- content_transformer(function (x , pattern )gsub(pattern, " ", x))
docs<- tm_map(docs, toSpace, "/") 
docs<- tm_map(docs, toSpace, "@") 
docs<- tm_map(docs,toSpace, "\\|")
docs<- tm_map(docs, content_transformer(tolower))
docs<- tm_map(docs, removeNumbers) 
docs<- tm_map(docs, removeWords, stopwords("english")) 
docs<- tm_map(docs, removeWords, c("blabla1", "blabla2")) 
docs<- tm_map(docs, removePunctuation) 
docs<- tm_map(docs, stripWhitespace) 
docs<- tm_map(docs, stemDocument)
dtm<- TermDocumentMatrix(docs) 
m <- as.matrix(dtm) 
write.csv(m, "Security_Raw_Dataset.csv")


 - - - - Weighthing method - - - -
 
 TF/IDF Weighting methods
setwd("C:/Users/Anam/Desktop/Final Experiment/Security/TDataset")
Dataset.file<- "C:/Users/Anam/Desktop/Final Experiment/Security/TDataset/Security.csv"
Dataset <- read.csv(Dataset.file,header=TRUE)
DS=Dataset[,3:ncol(Dataset)]
totaldoc=nrow(DS)
Terms=length(DS)
tf<-DS
idf<-log(nrow(DS)/colSums(DS))
tfidf<- DS
for(word in names(idf)){
tfidf[,word] <- tf[,word] * idf[word]
}
write.csv(tfidf, "Security_TFIDF.csv")


- - - Classification (XGBoost) - - - -

import nltk
nltk.download('averaged_perceptron_tagger')
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from numpy import sort
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from sklearn.feature_selection import SelectFromModel
import codecs
import matplotlib.pyplot as plt

np.random.seed(500)
Corpus = pd.read_csv("Security_Raw_Dataset_transpose_se.csv", encoding = "ISO-8859-1")

#Train_X, Train_Y = Corpus['body'], Corpus['label']
#TestData = pd.read_csv("testd1.csv",encoding='utf-8')
#Test_X, Test_Y = TestData['body'], TestData['label']

# Step - a : Remove blank rows if any.
cwitouth=Corpus.loc[:, Corpus.columns != 'Label']

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus.loc[:, Corpus.columns != 'Label'],Corpus['Label'],test_size=0.3)

#unvarient selection
bestfeatures = SelectKBest(score_func=chi2, k=100)
#bestfeatures1 = SelectKBest(score_func=chi2, k=1000)
fit = bestfeatures.fit(Train_X,Train_Y)
fit1 = bestfeatures.fit(Test_X,Test_Y)
X_important_train = fit.transform(Train_X)
X_important_test = fit.transform(Test_X)

# Classifier - Algorithm - XGBoost
modelc = XGBClassifier(objective='multi:softmax',
                      num_class=2,
                      n_estimators=100,
                      max_depth=2)
#bfx = bfx.reshape(190,1)
##Train_Y = Train_Y.reshape(190,1)
#bftx = bftx.reshape(47,1)

#eval_set = [(Train_X_Tfidf, Train_Y), (Test_X_Tfidf, Test_Y)]

#eval_set = [(Test_X_Tfidf, Test_Y)]

eval_set = [(X_important_train, Train_Y), (X_important_test, Test_Y)]
modelc.fit(X_important_train, Train_Y.ravel(), early_stopping_rounds=3, eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=False)

#history=modelc.fit(Train_X_Tfidf, Train_Y, verbose=True)
# make predictions for test data
y_pred = modelc.predict(X_important_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(Test_Y, predictions)
print("XGBoost Accuracy: %.2f%%" % (accuracy * 100.0))
#performance matrics
print(classification_report(Test_Y, modelc.predict(X_important_test)))
print(confusion_matrix(Test_Y, modelc.predict(X_important_test)))


# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(X_important_train,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(X_important_test)
NB_accuracy = accuracy_score(Test_Y, predictions_NB)
print("Naive Bayes Accuracy: %.2f%%" % (NB_accuracy * 100.0))
# confusion matrix and classification report(precision, recall, F1-score)
print(classification_report(Test_Y, Naive.predict(X_important_test)))
print(confusion_matrix(Test_Y, Naive.predict(X_important_test)))

def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len('Label'))
    plt.bar(index, accuracy)
    plt.xlabel('Genre', fontsize=5)
    plt.ylabel('No of Movies', fontsize=5)
    plt.xticks(index, 'Label', fontsize=5, rotation=30)
    plt.title('Market Share for Each Genre 1995-2017')
    plt.show()


results = modelc.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)

# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()

pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()


# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Test')
ax.legend()

pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()
 
