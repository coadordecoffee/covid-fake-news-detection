import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import csv
import tweepy
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

raw_df = pd.read_csv('CMU_MisCov19_dataset.csv')

fake_news_filter_array = ['conspiracy', 'fake cure', 'fake treatment',
                          'false fact or prevention', 'false public health response']

a1 = raw_df['annotation1'].unique()
print(a1)
print(fake_news_filter_array)

raw_df

fake_tweets_df = raw_df.loc[raw_df['annotation1'].isin(fake_news_filter_array)]

fake_tweets_df.drop(['status_created_at', 'annotation1',
                    'annotation2'], axis='columns', inplace=True)
fake_tweets_df

fake_tweets_df.to_csv("fake_tweets_id.csv", index=False)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

file = open("fake_tweets_df.csv", "w")
writer = csv.writer(file)

writer.writerow(["tweet", "class"])
sleepTime = 2

with open("fake_tweets_id.csv", "r") as tweet_ids:
    for tweet_id in tweet_ids:
        try:
            tweet = api.get_status(tweet_id)
            tweet_text = tweet._json['text']

            writer.writerow([tweet_text, 0])
            time.sleep(sleepTime)

        except tweepy.TweepError as e:
            excep_count += 1
            print(e)
            continue
        except StopIteration:
            break

file.close()

train_df = pd.read_csv("train_df.csv")

train_df

print("No of missing title\t:", train_df[train_df['tweetOG'].isna()].shape[0])
print("No of missing text\t:", train_df[train_df['tweetBR'].isna()].shape[0])
print("No of missing source\t:", train_df[train_df['class'].isna()].shape[0])

no_of_fakes = train_df.loc[train_df['class'] == 0].count()[0]
no_of_trues = train_df.loc[train_df['class'] == 1].count()[0]

print(no_of_fakes)
print(no_of_trues)

train_df.info(verbose=True)

train_df.isnull().sum()

nltk.download('stopwords')

stop_words = set(stopwords.words('portuguese'))

print(stop_words)


def clean(text):

    text = text.lower()

    #remove @ mentions
    text = re.sub(r'@[a-z0-9]+', '', text)

    # remove any kind of link
    text = re.sub(r'http\S+', '', text)

    # Removing numbers
    text = re.sub('[^a-zà-ü]+', ' ', text)

    word_tokens = word_tokenize(text)

    filtered_sentence = []
    for word_token in word_tokens:
        if word_token not in stop_words:
            filtered_sentence.append(word_token)

    text = (' '.join(filtered_sentence))
    return text


nltk.download('punkt')

print(train_df['tweetBR'][10])
print(train_df['tweetBR'][20])
print(train_df['tweetBR'][30])
print(train_df['tweetBR'][40])
print("apos limpeza...")

print(clean(train_df['tweetBR'][10]))
print(clean(train_df['tweetBR'][20]))
print(clean(train_df['tweetBR'][30]))
print(clean(train_df['tweetBR'][40]))

train_df['tweetBR'] = train_df['tweetBR'].apply(clean)

train_df['tweetBR']

train_df['tweetBR']

train_df.to_csv('clean_df.csv')

print(train_df.tweetBR.duplicated().sum())

train_df.drop_duplicates(subset=['tweetBR'], inplace=True)

print(train_df.tweetBR.duplicated().sum())

no_of_fakes = train_df.loc[train_df['class'] == 0].count()[0]
no_of_trues = train_df.loc[train_df['class'] == 1].count()[0]

print(no_of_fakes)
print(no_of_trues)

train_df

X = train_df['tweetBR']
y = train_df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, test_size=0.2, random_state=11)

X_train

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
array = X_train.toarray()
array

clf = MultinomialNB()
clf.fit(X_train, y_train)

X_test = vectorizer.transform(X_test)
X_test

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

predictions = clf.predict(X_test)
predictions

cm = confusion_matrix(y_test, predictions)
cm

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['FAKE', 'TRUE'], yticklabels=[
            'FAKE', 'TRUE'], cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

result = clf.predict(X_test)

print(accuracy_score(y_test, result))

print(classification_report(y_test, result))

# Exemplo de um tweet classificado como fake news
testSentence = "@DrJohnB2 Meu Deus... mais um elemento além das substâncias das w4cin4s. Óleo de silicone nas seringas. @SF_Moro @mariosabinof @DanielaLima_ @renataagostini @fabiarichter @GiselaSavioli @biodireito @consumidor_gov @JusticaGovBR PROCON? @JanainaDoBrasil @deltanmd @alexandregarcia"

testSentence = clean(testSentence)
vectorizedSentence = vectorizer.transform([testSentence]).toarray()
result = clf.predict(vectorizedSentence)

print(result[0])


def predict_tweet(tweet):

    cleanTweetText = clean(tweet)
    vectorizedSentence = vectorizer.transform([cleanTweetText]).toarray()
    prediction = clf.predict(vectorizedSentence)

    if prediction[0] == 0:
        label = 'Fake'
        labelPredict = clf.predict_proba(vectorizedSentence)[:, 0][0]
    else:
        label = 'True'
        labelPredict = clf.predict_proba(vectorizedSentence)[:, 1][0]

    return label, labelPredict


tweets_df = pd.read_csv('tweets.csv')

classification_array = []
confidence_array = []

for index, row in tweets_df.iterrows():
    classification, confidence = predict_tweet(row['tweet'])
    classification_array.append(classification)
    confidence_array.append(confidence)


df_predictions = pd.DataFrame(
    {'tweet': tweets_df['tweet'], 'classification': classification_array, 'confidence':  confidence_array})

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

df_predictions  # before

df = df_predictions.loc[df_predictions['classification'] == 'Fake']
df['tweet']


def get_top_n_words(corpus, n=None):
    vec2 = CountVectorizer().fit(corpus)
    bag_of_words = vec2.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec2.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


df['tweet']

common_words = get_top_n_words(df['tweet'], 50)
for word, freq in common_words:
    print(word, freq)

no_of_fakes = dfPredictions.loc[dfPredictions['classification'] == 'Fake'].count()[
    0]
no_of_trues = dfPredictions.loc[dfPredictions['classification'] == 'True'].count()[
    0]

print(no_of_fakes)
print(no_of_trues)
