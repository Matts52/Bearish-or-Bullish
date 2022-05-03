############ imports ##################

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess 
import spacy
import warnings
import pickle
from gensim.models import TfidfModel
from pprint import pprint
import pandas as pd
import dataset_utils
from xone import calendar
import pickle
from gensim.models import TfidfModel

warnings.filterwarnings("ignore")

############ Global Constants ###################
# irrelevant phrases to be filtered out
irrelevant = ['food & culture', 'football', "football's", "baseball", "baseball's", "nba", "mlb", "nfl", "mls",
              "nhl", "hockey", "hockey's", "yankees", "mets", "rangers", "soccer", "soccer's", "arts & entertainment",
              "arena", "jets'", "giants'", "sports", "arts & entertainment:", "heard & scene", "book", "books",
              "weekend journal", "off duty", "personal journal"]


############# Function Definitions #####################

def lemmatization(texts): # to keep with paper we will not remove any post tags
    # import the English spacy object as en
    en = spacy.load('en_core_web_sm', disable=["parser", "ner"])
    texts_out = []
    lemma_text = []
    # Lemmatize text
    count = 0 # for keeping track
    for article in texts:
        tokens = en(article)#[en(text) for text in article]
        [lemma_text.append(tok.lemma_) for tok in tokens] # lemmatize token if in allowed_postags
        final = " ".join(lemma_text)
        texts_out.append(final)
        lemma_text = []
        count+=1
        if count % 100 == 0:
            print(count)
    return texts_out

def gen_words(texts): # pass in your texts as an argument
    final = []
    for text in texts: # iterate over the texts
        new = gensim.utils.simple_preprocess(text, deacc=True) 
        final.append(new)
    return final # turns your texts into preprocessed, individual words

def make_bigrams(texts):
    # identify and reconsitute all of your bigrams
    return [bigram[doc] for doc in texts] #important to use square brackets for list comprehension

def business_dates(start, end):
    us_cal = calendar.USTradingCalendar()
    kw = dict(start=start, end=end)
    return pd.bdate_range(**kw).drop(us_cal.holidays(**kw))

def remove_stopwords(corpus):
    print("getting ready")
    en = spacy.load('en_core_web_sm')
    stopwords = en.Defaults.stop_words
    texts_out = []
    article = [line for line in corpus]
    split_articles = [sentence.split(" ") for sentence in article]
    count = 0
    for item in split_articles:
        tokens_filtered = [token for token in item if token not in stopwords]
        final = (" ").join(tokens_filtered)
        texts_out.append(final)
        count += 1
        if count % 100 == 0:
            print(count)
    return texts_out

def gen_words(texts): # pass in your texts as an argument
    final = []
    count = 0
    for text in texts: # iterate over the texts
        new = gensim.utils.simple_preprocess(text, deacc=True) 
        final.append(new)
        count+=1
        if count % 1000 == 0:
            print(count)
    return final


########## Cleaning Script ##################

# read in the replication articles
dfs = []
for fn in ['WSJ_Art_Rep_Strings_Even_First15_2006_2016.csv', 'WSJ_Art_Rep_Strings_Evens_Last25_2006_2016.csv', \
          'WSJ_Art_Strings_Rep_Odds_First15_2007_2017.csv', 'WSJ_Art_Strings_Rep_Odds_Last25_2007_2017.csv']:
    dfs.append(pd.read_csv(fn, encoding = "ISO-8859-1"))

# create the replication DataFrame
df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset = ['text'])
df = df.drop_duplicates()
rep_df = df.reset_index().filter(['link', 'text'])

# Grab the article metadata for every article
dfs = []
for year in range(2007,2018):
    if year == 2008:
        continue
    fn = "WSJ_Art_Links_" + str(year) +".csv"
    dfs.append(pd.read_csv(fn, encoding = "ISO-8859-1"))
dfs.append(pd.read_csv('WSJ_Art_Links_2014_part_2.csv', encoding = "ISO-8859-1"))
dfs.append(pd.read_csv('WSJ_Art_Links.csv', encoding="ISO-8859-1"))
print(dfs)
links = pd.concat(dfs, ignore_index=True)

# Select only article metadata of interest
colsToKeep = ["year", "month", "day", "link", "title", "author"]
links = links[colsToKeep]
links.tail()
links['year'].value_counts(normalize=True) * 100

# merge the replication dataframe with corresponding article metadata
merged = rep_df.merge(links, how='inner', on='link')
merged['datetime'] = pd.to_datetime(merged[['year', 'month', 'day']])
merged = merged.sort_values('datetime')
merged['year'].value_counts(normalize=True) * 100

# filter and replace all non-alphabetic characters
merged['alpha'] = merged['text'].str.replace(r'[^A-Za-z ]+', '', regex=True)
merged['alpha'] = merged['alpha'].str.lower()

#select only articles which happened on days when the market was open
business_dates = business_dates(start='2006-01-01', end='2017-12-31')
cleaning = merged[merged.datetime.isin(business_dates)]

# eliminate 'stub' articles of less than 100 tokens
cleaning['word_count'] = cleaning['text'].apply(lambda x: len(x.split()))
condition = cleaning['word_count'] > 99
temp = cleaning[condition]
cleaning = cleaning.reset_index(drop=True)

# remove stopwords and website related tokens
cleaning['alpha'] = cleaning['alpha'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)

# lemmatize article tokens and save to output file
lemmatized = lemmatization(cleaning['alpha'])
with open('WSJ_lemmatized_v2.pkl', 'wb') as f:
    pickle.dump(lemmatized, f)

# remove stop words from all articles in the corpus -> add this to our cleaning DataFrame
cleaned_text = remove_stopwords(lemmatized)
cleaned_text[0][0:20]
cleaning['lemmatized_and_stops'] = cleaned_text

# remove tokens of less than 3 characters
cleaning['lemmatized_and_stops'] = cleaning['lemmatized_and_stops'].str.replace(r'\b\w{1,2}\b', '', regex=True)

# generate words from our gathered article texts
preprocessed = gen_words(cleaned_text)

# create our bigrams and incorporate them into our database
bigram_phrases = gensim.models.Phrases(preprocessed, min_count=5, threshold=200)
bigram = gensim.models.phrases.Phraser(bigram_phrases)
data_bigrams = make_bigrams(preprocessed)
cleaning['data_bigrams'] = data_bigrams

# create the Bag of Words model of the article texts
id2word = corpora.Dictionary(data_bigrams)
print("There are {num} words in the corpus before pruning".format(num = (len(id2word))))
id2word.filter_extremes(no_above=8)
print("There are {num} words in the corpus after pruning".format(num = (len(id2word))))
texts = data_bigrams
corpus = [id2word.doc2bow(text) for text in texts] 

# append our finished corpus to the dataset and save it to a file
cleaned_data = cleaning
cleaned_data['corpus'] = corpus
cleaned_data
with open('/Users/sam/PycharmProjects/Eco2460/WSJ_id2word.pkl', 'wb') as f:
    pickle.dump(id2word, f)

# export our finished clean dataset
cleaned_data.to_pickle('/Users/sam/PycharmProjects/Eco2460/cleaned_WSJ_data.pkl')




























