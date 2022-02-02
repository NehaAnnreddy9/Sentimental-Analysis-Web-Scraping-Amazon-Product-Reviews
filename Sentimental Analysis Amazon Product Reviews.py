import pandas as pd
dfn=pd.read_csv("amazon-reviews-scrape.csv")
df7=dfn
dfn.head()

dfnn=dfn["content"]
dfnn

import textblob
tokenized_comm=dfnn.apply(lambda x: x.split())
tokenized_comm.head()

from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_comm = tokenized_comm.apply(lambda x: [stemmer.stem(i) for i in x])
tokenized_comm.head()

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
dfnn = dfnn.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
dfnn.head()

df5["Text"]=dfnn
df5['sentiment'] = dfnn.apply(lambda x: TextBlob(x).sentiment[0] )
df6=df5[['Text','sentiment']]

df6.head()
def sentval(sentiment):
    if(sentiment<0):
        return "negative"
    if(sentiment>0):
        return "positive"
    else:
        return "neutral"

df6['review_score']=df6["sentiment"].apply(sentval)
df6
df7['review_score']=df6['review_score']

cnt=df6["review_score"].value_counts()
cnt

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.figure(figsize=(12,8))
cnt=df6['review_score'].value_counts()
sns.barplot(x=cnt.index,y=cnt.values,alpha=0.8)
plt.xlabel("review_score")
plt.ylabel("No of occurences")
plt.title("barplot of sentiment")
plt.show()

f, ax = plt.subplots( figsize=(16,8))
colors = ["#00FF00", "#FF0000","#000000"]
labels ="Positive", "Negative","Neutral"

plt.suptitle('Pie chart ', fontsize=20)

df6["review_score"].value_counts().plot.pie(explode=[0,0.05,0.05], autopct='%1.2f%%', ax=ax, shadow=True, colors=colors,
                       labels=labels, fontsize=12, startangle=70)



ax.set_xlabel('% of positive, negative and neutral reviews', fontsize=14)


from wordcloud import WordCloud,STOPWORDS


train_pos = df7[ df7['review_score'] == 'positive']
train_pos = train_pos['content']
train_neg = df7[ df7['review_score'] == 'negative']
train_neg = train_neg['content']

def wordcloud_draw(data, color = 'black'):
  words = ' '.join(data)
  cleaned_word = " ".join([word for word in words.split()
              if 'http' not in word
                and not word.startswith('@')
                and not word.startswith('#')
                and word != 'RT'
              ])
  wordcloud = WordCloud(stopwords=STOPWORDS,
           background_color=color,
           width=2500,
           height=2000
           ).generate(cleaned_word)
  plt.figure(1,figsize=(13, 13))
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()


