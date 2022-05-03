
############# Imports ##########################

import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

################ Function Definitions ########################

def get_article_sentiment(article, pipeline):
    try:
        sentiments = []
        sentences = nltk.tokenize.sent_tokenize(article)[:5]
        for s in sentences:
            sentiment_dict = pipeline(s)
            label = sentiment_dict[0]['label']
            sentiments.append(label)
    except:
        # default to neutral sentiment if an error is thrown
        sentiments = ['NEUTRAL', 'NEUTRAL', 'NEUTRAL', 'NEUTRAL', 'NEUTRAL']
        
    return sentiments


def get_all_sentiments(corpus,pipeline):
    all_sentiments = []
    count = 0
    for article in corpus:
        sentiment = get_article_sentiment(article,pipeline)
        all_sentiments.append(sentiment)
        count+=1
        if count%100 == 0:
            print(sentiment)
            print(nltk.tokenize.sent_tokenize(article)[:5])
            print(count)

    return all_sentiments


################ Sentiment Building Script ###################3

# Load in the articles
corpus_load = pd.read_csv(r"")

# Load in the transformer
tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

#predict the sentiment of each article we have loaded
corpus = corpus_load['text']
start = 0
end = len(corpus)

save_list_2 = get_all_sentiments(corpus[start:end], sentiment_pipeline)
save_file_2 = pd.DataFrame(save_list_2)
save_file_2.to_csv('C:/Users/senic/OneDrive/Desktop/Masters/WINTER_2021/ECO2460/Term_Paper/Code/IMPORTANT_SENTIMENT_2.csv')















