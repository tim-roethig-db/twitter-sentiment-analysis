import re, unicodedata
import inflect
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from textblob import TextBlob

def remove_link(tweets):
    new_tweets = []
    for tweet in tweets:
        new_tweet = re.sub(r"http\S+", "", tweet)
        new_tweets.append(new_tweet)
    return new_tweets

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def spellcorection(words):
    new_words = []
    for word in words:
        corrected_word = TextBlob(word).correct()
        new_words.append(corrected_word)
    return new_words

def removeElements(text, k):
    text = [w for w in text if text.count(w) > k]
    return text


def retrieveFeatures(df, word_frequency):
    tweets = df['tweet_content'].tolist()
    print(tweets)

    print('---remove link---')
    tweets = remove_link(tweets)

    tweets = str(tweets).strip('[]')
    print(tweets)

    print('---Tokenize---')
    tweets = word_tokenize(tweets)
    print(len(tweets))
    print(tweets[:100])

    # to lowercase
    tweets = to_lowercase(tweets)

    print('---remove punctuation---')
    tweets = remove_punctuation(tweets)
    print(len(tweets))
    print(tweets[:100])

    print('---remove numbers---')
    tweets = [w for w in tweets if not any(c.isdigit() for c in w)]
    print(len(tweets))
    print(tweets[:100])

    print('---remove Stopwords and short words 1---')
    tweets = [w for w in tweets if w not in stopwords.words('english') and len(w) > 1]
    print(len(tweets))
    print(tweets[:100])

    print('---stemm words---')
    stemmer = PorterStemmer()
    tweets = [stemmer.stem(w) for w in tweets]
    print(len(tweets))
    print(tweets[:100])
    '''
    print('---lemmatize words---')
    lemmatizer = WordNetLemmatizer()
    tweets = [lemmatizer.lemmatize(w) for w in tweets]
    print(len(tweets))
    print(tweets[:100])
    '''
    print('---remove Stopwords and short words 2---')
    tweets = [w for w in tweets if w not in stopwords.words('english') and len(w) > 1]
    print(len(tweets))
    print(tweets[:100])

    print('---remove infrequent words---')
    tweets = removeElements(tweets, word_frequency)
    print(len(tweets))
    print(tweets[:100])

    print('---create features---')
    features = list(set(tweets))
    print(len(features))
    print(features)

    return features

def retrieveWordsFromTweet(tweet):
    # Tokenize
    tweet = word_tokenize(tweet)

    # to lowercase
    tweet = to_lowercase(tweet)

    # remove punctaition
    tweet = remove_punctuation(tweet)

    # remove Stopwords and short words 1
    tweet = [w for w in tweet if w not in stopwords.words('english') and len(w) > 1]

    # stemm words
    stemmer = PorterStemmer()
    tweet = [stemmer.stem(w) for w in tweet]
    '''
    # lemmatize words
    lemmatizer = WordNetLemmatizer()
    tweet = [lemmatizer.lemmatize(w) for w in tweet]
    '''

    return tweet

def createDataset(df, features):
    for feature in features:
        df[feature] = [0]*len(df)
    for index in range(len(df)):
        #print(df.iloc[index])
        words = retrieveWordsFromTweet(df.iloc[index]['tweet_content'])
        #print(words)
        for word in words:
            if word in features:
                df.at[index, word] += 1
    df.pop('tweet_content')
    return df


def prepareData(word_frequency):
    '''
    df = pd.read_csv('train1600000.csv', delimiter=',', encoding = "ISO-8859-1", names=['sentiment', 'id', 'date', 'flag', 'user', 'tweet_content'])
    df = df.drop(columns=['id', 'date', 'flag', 'user'])

    df = pd.read_csv('train.csv', delimiter=',', encoding="ISO-8859-1", names=['id', 'sentiment', 'tweet_content'])
    df = df.drop(columns=['id'])
    '''
    df = pd.read_csv('downloadedB.csv', delimiter='\t', encoding="ISO-8859-1", names=['id1', 'id2', 'sentiment', 'tweet_content'])
    df = df.drop(columns=['id1', 'id2'])
    df = df.drop(df[(df['tweet_content'] == 'Not Available')].index)
    df = df.drop(df[(df['sentiment'] == 'neutral')].index)
    df.loc[(df.sentiment == 'positive'), 'sentiment'] = 1
    df.loc[(df.sentiment == 'neutral'), 'sentiment'] = 0
    df.loc[(df.sentiment == 'negative'), 'sentiment'] = -1

    print(df)
    df = df.sample(frac=1).reset_index(drop=True)
    #df = df.iloc[:500]

    features = retrieveFeatures(df, word_frequency)

    df = createDataset(df, features)
    print(df)

    train, test = train_test_split(df, test_size=0.2)

    train_Y = train.pop('sentiment').values.tolist()
    train_X = train.values.tolist()
    test_Y = test.pop('sentiment').values.tolist()
    test_X = test.values.tolist()

    return train_X, train_Y, test_X, test_Y

#prepareData(1250)