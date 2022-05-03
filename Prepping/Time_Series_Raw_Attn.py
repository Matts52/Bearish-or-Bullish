
############ Imports ##################
import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
from datetime import datetime
import pandas_datareader as pdr
from gensim import models

warnings.filterwarnings("ignore")

############# Function Definitions ###########

def get_fred_data(param_list, start_date, end_date):
    df = pdr.DataReader(param_list, 'fred', start_date, end_date)
    return df


############# Relative Attention Weighting Script ###################

# read in the BOW representation and our generated LDA model
df = pd.read_pickle('C:/Users/senic/OneDrive/Desktop/Masters/WINTER_2021/ECO2460/Term_Paper/Code/Replication_Data/cleaned_WSJ_data_V2.pkl')
df_id = pd.read_pickle('C:/Users/senic/OneDrive/Desktop/Masters/WINTER_2021/ECO2460/Term_Paper/Code/Replication_Data/WSJ_id2word_V2.pkl')
lda_model = models.LdaModel.load('C:/Users/senic/OneDrive/Desktop/Masters/WINTER_2021/ECO2460/Term_Paper/Code/models/50_model_slow_V5.model')


# aggregate attention by day and weight it according to articles seen during that day

topic_weights = []
dates = []
years = [i for i in range(2006, 2018)]
months = [i for i in range(1,13)]
days = [i for i in range(1,33)]
tops = len(lda_model.get_topics())
corpus = df['corpus'].tolist()

c = 0
l = 365 * 12
#loop thru each month and year
for y in years:
    for m in months:
        for d in days:
            #check if this is a weekend
            try:
                if datetime(y, m, d).weekday() > 4:
                    c+=1
                    continue
            except:
                c+=1
                continue
            
            #count the documents in teh current day and accumulate the predicted weight of each topic within the documents
            doc_count = 0
            acc_weight = np.zeros(tops)
            temp_df = df.loc[(df['year'] == y) & (df['month'] == m) & (df['day'] == d), ['title']]
            
            for index, row in temp_df.iterrows():
                new_weight = lda_model[corpus[index]]
                for weight in new_weight:
                    acc_weight[weight[0]] += weight[1]
                doc_count += 1
            #find the relative weight of each topic during the day
            if doc_count != 0:
                acc_weight = acc_weight / doc_count
                topic_weights.append(acc_weight)
                
                dd, mm = str(d), str(m)
                if d < 10: dd = '0'+dd
                if m < 10: mm = '0'+mm
                
                dates.append(str(y)+'-'+str(mm)+'-'+str(dd))
            #do not append days which had no documents (holidays)
            else:
                pass
            c += 1
            if c % 200 == 0:
                print(round(c/l,2) * 100, '%')


topic_weights = np.array(topic_weights)
dates = np.array(dates)

#plot each of the time-series relative weights of the topics
#again using months so that we can have a representative number of documents while avoiding getting overly granular using daily data
plt.rcParams["figure.figsize"] = (50,10)
for i in range(44, 45): #tops):
    plt.xlabel('Date', fontsize=30)
    plt.ylabel('Weight', fontsize=30)
    plt.title('Topic '+str(i), fontsize=40)
    plt.tick_params(axis='both',labelsize=25)
    plt.xticks(rotation=45, ha='right')
    plt.plot(dates, topic_weights[:,i])
    xticks = plt.gca().xaxis.get_major_ticks()
    for i in range(len(xticks)):
        if i % 100 != 0:
            xticks[i].set_visible(False)
    plt.show()
    
# read in the independent FRED Market Volatility Daily Metric
series = 'VIXCLS'
fred_data = get_fred_data(param_list=[series], start_date='2006-01-01', end_date='2017-12-31')
fred_data['VIXCLS'][0] = 0.0
ind_data = fred_data

# only keep business days, as that is when the market is open

index_growth = np.array(ind_data['VIXCLS'].tolist())
ind_dates = ind_data.index.tolist()
ind_dates = [date_obj.strftime('%Y%m%d') for date_obj in ind_dates]

inds = []
for i in range(0, len(dates)): 
    if dates[i].replace('-','') not in ind_dates:
        print(dates[i])
        inds.append(i)

dates_clean = [i for j, i in enumerate(dates) if j not in inds]        
rebuild_weights = [i for j, i in enumerate(topic_weights) if j not in inds]

c = 0
for i in range(0, len(ind_dates)):
    if ind_dates[i][0:4]+'-'+ind_dates[i][4:6]+'-'+ind_dates[i][6:] not in dates:
        ind_data.drop(index=ind_dates[i][0:4]+'-'+ind_dates[i][4:6]+'-'+ind_dates[i][6:], inplace = True)
    else:    
        c += 1


# Rebuild our full daily topic-weight matrix
vixcls = np.array(ind_data['VIXCLS'].tolist())
dates_1 = np.array(dates_clean)
rebuild_weights = pd.DataFrame(rebuild_weights)
rebuild_weights['VIXCLS'] = vixcls
rebuild_weights['date'] = dates_1
raw_attn = rebuild_weights
rebuild_weights

# save our topic-weight matrix to a file for later use during prediction
raw_attn.to_csv('C:/Users/senic/OneDrive/Desktop/Masters/WINTER_2021/ECO2460/Term_Paper/Code/Replication_Data/var_input_raw_att_50_V5.csv')













