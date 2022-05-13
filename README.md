# Bearish or Bullish? A Predictive Sentiment-Weighted Attention Analysis of the Wall Street Journal by Matthew Senick and Samuel Schwartzbein

*Disclaimer: All code within is equally attributed to my paper partner and all website links for scraping as well as file import links are redacted to protect the privacy of the owners*

Read the paper [here](https://www.matthewsenick.com/Bearish_or_Bullish.pdf)

### Overview

This project serves as a complete analysis of the predictive and explanatory power of the language and sentiment utilized by the Wall Street Journal (WSJ). Inspiration for the project was Bybee et al.'s paper "Business News and Business Cycles." [link](https://www.nber.org/papers/w29344) who run a similar analysis, however only consider attention and utilize a Vector Autoregressive model to predict the market in the end. The flow of this code is as follows: Firstly, we scrape a database of WSJ articles for article contents from 2006 until 2017. Secondly, we clean the collected articles, build an LDA model, collect proxies for per-article sentiment, and create a time-series sentiment-weighted topic attention metric. Thirdly, the prepped data is analyzed by utilizing a variety of market fluctuation indicators as response variables and our developed sentiment-weighted topic attentions as predictors. Additionally, we explore authorship bias of the publication by predicting the most prolific articles based on articles texts. In the end, this results in a Lasso regression using a prespecified number of most informative topics to predict directional change in specified market fluctuation indicators.

### Scraping

In this stage, we gather the necessary data for our resulting analyses. Initially, we utilize a web-scraper to create a dataset of article links by scraping a directory of WSJ articles. Using this dataset of article links, a second web-scraper was developed to grab article contents as well as various pieces of meta-data, such as the author, date, etc. Upon completion of scraping the years 2006-2017, approximately 100,000 articles were scraped.

### Prepping

Having collected article contents, we are then tasked with cleaning the data. We follow almost exactly the same cleaning steps as Bybee et al. except for only being able to drop articles based on keywords and not by the articles section. Once cleaned, we have a tokenized Bag-of-Words model of each article in the database, which is fed to a Gensim LDA topic model to represent and cluster tokens and themes in the article texts. Alongside this, we generated a sentiment score of -1 or 1 for each article by analyzing the first 5 sentences of the article for negative and positive articles respectively using the RoBERTa sentient transformer. Given these items, we are ready to generate a day-to-day sentiment-weight time series dataset of topic attention featuring the pre-selected number of topic. We do this by summing normalized per-topic attention for all articles in a given day and introducing article sentiment as a multiplicative constant to a given article. A highly positive value for a given day means that this topic was very important and also very positive during the day and vice versa. 

### Analysis

First off, as a check for bias, we created a Naive Bayesian model to predict article authorship from verbiage of a given article. As shown by the results, bias may be a problem, but when taking a closer look at the most important predicitive words, this concern disappears as they are typically content type words rather than words of heavy sentimental value.

Now that we have created our sentiment-weighted topic attention predictors, we introduce numerous different market fluctuation indicators. However, in our analyses, of most importance was the Market Volatility indicator from FRED. By introducing sentiment to the model, we were able to observe slightly better in-sample and out-of-sample R^2 and MSE than the original paper by Bybeeet al. 

As a check, we also analyzed the sentiment polarity of all topics. As observed, we can easily pick out which topics were important for swaying negative market fluctuation movements and which were important for influencing positive market fluctuation movements. 



