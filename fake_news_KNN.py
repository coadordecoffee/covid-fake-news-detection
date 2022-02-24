import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# df = pd.read_csv("train_df_covid.csv", sep=";")
train_df = pd.read_csv("train_df.csv")

train_df

print("No of missing title\t:", train_df[train_df['tweetOG'].isna()].shape[0])
print("No of missing text\t:", train_df[train_df['tweetBR'].isna()].shape[0])
print("No of missing source\t:",
      train_df[train_df['class_knn'].isna()].shape[0])

no_of_fakes = train_df.loc[train_df['class'] == 0].count()[0]
no_of_trues = train_df.loc[train_df['class'] == 1].count()[0]

print("fakes: ", no_of_fakes)
print("trues: ", no_of_trues)

train_df.info(verbose=True)

nltk.download('stopwords')

stop_words = set(stopwords.words('portuguese'))

print(stop_words)


def clean(text):
      text = text.lower()

  # remove @ mentions
  text = re.sub(r'@[a-z0-9]+','',text)
  
  # remove any kind of link
  text= re.sub(r'http\S+', '', text) 

  # Removing numbers
  text = re.sub('[^a-zà-ü]+',' ',text) 

  word_tokens = word_tokenize(text)

  filtered_sentence = []
  for word_token in word_tokens:
      if word_token not in stop_words:
          filtered_sentence.append(word_token)
  
  text = (' '.join(filtered_sentence))

nltk.download('punkt')

print(train_df['tweetBR'][1])
print(train_df['tweetBR'][2])
print(train_df['tweetBR'][3])
print(train_df['tweetBR'][4])

print("após limpeza...")

print(clean(train_df['tweetBR'][1]))
print(clean(train_df['tweetBR'][2]))
print(clean(train_df['tweetBR'][3]))
print(clean(train_df['tweetBR'][4]))

train_df['tweetBR'] 

train_df['tweetBR'] = train_df['tweetBR'].apply(clean)

train_df['tweetBR']

print(train_df.tweetBR.duplicated().sum())

train_df.drop_duplicates(subset=['tweetBR'], inplace=True)

print(train_df.tweetBR.duplicated().sum())

print(train_df['tweetBR'])

train_df.info(verbose=True)

no_of_fakes = train_df.loc[train_df['class'] == 0].count()[0]
no_of_trues = train_df.loc[train_df['class'] == 1].count()[0]

print(no_of_fakes)
print(no_of_trues)

X = train_df['tweetBR']
y = train_df['class_knn']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=11)

X_train

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
array = X_train.toarray()
array

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# Creating object that corresponds to model with specified params
model = KNeighborsClassifier()
# Identifying correct number of neighbors
param_grid = {'n_neighbors': np.arange(1,25)}
knn_gscv = GridSearchCV(model, param_grid)
knn_gscv.fit(X_train, y_train)

# Check value of best precision
knn_gscv.best_params_

knn_gscv.best_score_

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

X_test = vectorizer.transform(X_test)
X_test

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

# Test data preview
predictions = model.predict(X_test)
predictions

cm = confusion_matrix(y_test, predictions)
cm

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['FAKE', 'TRUE'], yticklabels=['FAKE', 'TRUE'], cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

result = model.predict(X_test) 
print("precisão: ", accuracy_score(y_test, result)) 
print (classification_report(y_test, result))

# Example
testSentence = "@DrJohnB2 Meu Deus... mais um elemento além das substâncias das w4cin4s. Óleo de silicone nas seringas. @SF_Moro @mariosabinof @DanielaLima_ @renataagostini @fabiarichter @GiselaSavioli @biodireito @consumidor_gov @JusticaGovBR PROCON? @JanainaDoBrasil @deltanmd @alexandregarcia"

testSentence = clean(testSentence)
vectorizedSentence = vectorizer.transform([testSentence]).toarray()
result = model.predict(vectorizedSentence)

print(result[0])

def predictTweet(tweet):
    
  cleanTweetText = clean(tweet)
  vectorizedSentence = vectorizer.transform([cleanTweetText]).toarray()
  prediction = model.predict(vectorizedSentence)
  
  if prediction[0] == 0:
    label = 'Fake'
    labelPredict = model.predict_proba(vectorizedSentence)[:,1][0]
  else:
    label = 'True'
    labelPredict = model.predict_proba(vectorizedSentence)[:,0][0]

  return label, labelPredict

tweets_df = pd.read_csv('tweets.csv')

classification_array = []
confidence_array = []

for index, row in tweets_df.iterrows():
    classification, confidence = predictTweet(row['tweet'])
    classification_array.append(classification)
    confidence_array.append(confidence)

print(len(classification_array))
print(len(confidence_array))

dfPredictions = pd.DataFrame({'tweet': tweets_df['tweet'], 'classification': classification_array, 'confidence':  confidence_array})

 # before
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
dfPredictions

# after
dfPredictionsOrd = dfPredictions.sort_values('classification', inplace=False)
dfPredictionsOrdOld = dfPredictionsOrd
dfPredictionsOrd = dfPredictionsOrd.drop('confidence', axis=1)
dfPredictionsOrd

no_of_fakes = dfPredictions.loc[dfPredictions['classification'] == 'Fake'].count()[0]
no_of_trues = dfPredictions.loc[dfPredictions['classification'] == 'True'].count()[0]

print("number of fakes in test: ", no_of_fakes)
print("number of trues in test: ", no_of_trues)