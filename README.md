
## 1. Importing Libraries


```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import pandas as pd
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud, ImageColorGenerator
import seaborn as sns
from nltk.util import ngrams

plt.style.use('ggplot')
```

    C:\Users\HPHP\Anaconda3\lib\site-packages\nltk\twitter\__init__.py:20: UserWarning: The twython library has not been installed. Some functionality from the twitter package will not be available.
      warnings.warn("The twython library has not been installed. "
    


```python
pd.options.display.max_colwidth=200
```

## 2. Initializing SentimentIntensityAnalyzer


```python
sentiment = SentimentIntensityAnalyzer()
```


```python
def print_sentiment_scores(sentence):
    snt = sentiment.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(snt)))
```

## 3. Reading the data from CSV file


```python
data = pd.read_csv('Tweets.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>airline_sentiment</th>
      <th>airline_sentiment_confidence</th>
      <th>negativereason</th>
      <th>negativereason_confidence</th>
      <th>airline</th>
      <th>airline_sentiment_gold</th>
      <th>name</th>
      <th>negativereason_gold</th>
      <th>retweet_count</th>
      <th>text</th>
      <th>tweet_coord</th>
      <th>tweet_created</th>
      <th>tweet_location</th>
      <th>user_timezone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>570306133677760000</td>
      <td>neutral</td>
      <td>1.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>cairdin</td>
      <td>NaN</td>
      <td>0</td>
      <td>@VirginAmerica What @dhepburn said.</td>
      <td>NaN</td>
      <td>2/24/2015 11:35</td>
      <td>NaN</td>
      <td>Eastern Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>570301130888122000</td>
      <td>positive</td>
      <td>0.3486</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>jnardino</td>
      <td>NaN</td>
      <td>0</td>
      <td>@VirginAmerica plus you've added commercials t...</td>
      <td>NaN</td>
      <td>2/24/2015 11:15</td>
      <td>NaN</td>
      <td>Pacific Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>570301083672813000</td>
      <td>neutral</td>
      <td>0.6837</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>yvonnalynn</td>
      <td>NaN</td>
      <td>0</td>
      <td>@VirginAmerica I didn't today... Must mean I n...</td>
      <td>NaN</td>
      <td>2/24/2015 11:15</td>
      <td>Lets Play</td>
      <td>Central Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>570301031407624000</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Bad Flight</td>
      <td>0.7033</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>jnardino</td>
      <td>NaN</td>
      <td>0</td>
      <td>@VirginAmerica it's really aggressive to blast...</td>
      <td>NaN</td>
      <td>2/24/2015 11:15</td>
      <td>NaN</td>
      <td>Pacific Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>570300817074462000</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Can't Tell</td>
      <td>1.0000</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>jnardino</td>
      <td>NaN</td>
      <td>0</td>
      <td>@VirginAmerica and it's a really big bad thing...</td>
      <td>NaN</td>
      <td>2/24/2015 11:14</td>
      <td>NaN</td>
      <td>Pacific Time (US &amp; Canada)</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (14640, 15)




```python
tweets = pd.DataFrame(data['text'][:1000])
tweets.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@VirginAmerica What @dhepburn said.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@VirginAmerica plus you've added commercials to the experience... tacky.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@VirginAmerica I didn't today... Must mean I need to take another trip!</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@VirginAmerica it's really aggressive to blast obnoxious "entertainment" in your guests' faces &amp;amp; they have little recourse</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@VirginAmerica and it's a really big bad thing about it</td>
    </tr>
    <tr>
      <th>5</th>
      <td>@VirginAmerica seriously would pay $30 a flight for seats that didn't have this playing.\nit's really the only bad thing about flying VA</td>
    </tr>
    <tr>
      <th>6</th>
      <td>@VirginAmerica yes, nearly every time I fly VX this “ear worm” won’t go away :)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>@VirginAmerica Really missed a prime opportunity for Men Without Hats parody, there. https://t.co/mWpG7grEZP</td>
    </tr>
    <tr>
      <th>8</th>
      <td>@virginamerica Well, I didn't…but NOW I DO! :-D</td>
    </tr>
    <tr>
      <th>9</th>
      <td>@VirginAmerica it was amazing, and arrived an hour early. You're too good to me.</td>
    </tr>
  </tbody>
</table>
</div>



### Looks like there are lot of references to other twitter handles and hyperlinks. Let's remove these using regular expressions.


```python
data['text'] = data['text'].apply(lambda x: ' '.join(re.sub("(@[A-Za-z0-9]+)|(\w+:\/\/\S+)"," ",x).split()))
```


```python
data['text'].head(10) # Now the tweets are looking cleaner
```




    0                                                                                                                  What said.
    1                                                                   plus you've added commercials to the experience... tacky.
    2                                                                    I didn't today... Must mean I need to take another trip!
    3             it's really aggressive to blast obnoxious "entertainment" in your guests' faces &amp; they have little recourse
    4                                                                                    and it's a really big bad thing about it
    5    seriously would pay $30 a flight for seats that didn't have this playing. it's really the only bad thing about flying VA
    6                                                            yes, nearly every time I fly VX this “ear worm” won’t go away :)
    7                                                       Really missed a prime opportunity for Men Without Hats parody, there.
    8                                                                                            Well, I didn't…but NOW I DO! :-D
    9                                                           it was amazing, and arrived an hour early. You're too good to me.
    Name: text, dtype: object



## We can not remove punctuation, because it removes emoticons as well. Emoticons play a role in deciding Polarity strength.

## 4. Applying Vader on a sample sentence


```python
#Using the reviews, accessing the first row and specifying the compound measure. We are not running on all rows just yet.

print('polarity score-compound: ', sentiment.polarity_scores('It was a good experience :)')['compound'])
```

    polarity score-compound:  0.7096
    

## 5. Creating "Compound Polarity Score" for each tweet


```python
data['polarity_score'] = data['text'].apply(lambda x: sentiment.polarity_scores(str(x))['compound'])
```


```python
data[['text','polarity_score']][0:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>polarity_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What said.</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>plus you've added commercials to the experience... tacky.</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I didn't today... Must mean I need to take another trip!</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>it's really aggressive to blast obnoxious "entertainment" in your guests' faces &amp;amp; they have little recourse</td>
      <td>-0.5984</td>
    </tr>
    <tr>
      <th>4</th>
      <td>and it's a really big bad thing about it</td>
      <td>-0.5829</td>
    </tr>
  </tbody>
</table>
</div>



Let us add one more column that will store the sentiment in words for each column of reviews

## 6. Creating 2 new columns to store Vader's sentiment


```python
data['sentiment']=''
data['polarity']=''
```

## 7. Binning the "Compound Polarity Score" into 5 sentiments

#### Intervals

- (-1, -0.5) : 1, V.Negative
- (-0.5, 0) : 2, Negative
- (0) : 3, Neutral
- (0, 0.5) : 4, Positive
- (0.5, 1) : 5, V.Positive


```python
# Creating Sentiment labels with 5 classes
data.loc[(data.polarity_score<=1) & (data.polarity_score>=0.5),'sentiment']='V.Positive'
data.loc[(data.polarity_score<0.5) & (data.polarity_score>0),'sentiment']='Positive'
data.loc[(data.polarity_score==0),'sentiment']='Neutral'
data.loc[(data.polarity_score<0) & (data.polarity_score>=-0.5),'sentiment']='Negative'
data.loc[(data.polarity_score<-0.5) & (data.polarity_score>=-1),'sentiment']='V.Negative'
```


```python
# Creating Polarity labels with 3 classes
data.loc[(data.polarity_score<=1) & (data.polarity_score>0),'polarity']='positive'
data.loc[(data.polarity_score==0),'polarity']='neutral'
data.loc[(data.polarity_score<0) & (data.polarity_score>=-1),'polarity']='negative'
```

## 8. Bar plot of Vader sentiments


```python
%matplotlib inline
plt.figure(figsize=(18,8))
plt.xlabel("Sentiment")
plt.ylabel("Count of Tweets")
data.sentiment.value_counts().plot(kind='bar', title="Sentiment Distribution - All Tweets")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d9606b72b0>




![png](output_27_1.png)


## 9. Bar plot of Airline Sentiments


```python
plt.figure(figsize=(18,8))
plt.xlabel("Airline Labelled Sentiment")
plt.ylabel("Count of Tweets")
data.airline_sentiment.value_counts().plot(kind='bar', title="Airline Labelled Sentiment Distribution - All Tweets")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d960221a58>




![png](output_29_1.png)



```python
plt.figure(figsize=(18,8))
plt.xlabel("Polarity")
plt.ylabel("Count of Tweets")
data.polarity.value_counts().plot(kind='bar', title="Polarity Distribution - All Tweets")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d900e45f60>




![png](output_30_1.png)


### From above 2 plots, we can see the Vader analyzer labelled more tweets as positive compared to airline sentiments.

## 10. Accuracy of Vader's Sentiment compared to Airline labelled sentiments


```python
#find accuracy
matched_sent = data[data['airline_sentiment'] == data['polarity']]
```


```python
accuracy = (len(matched_sent)/data.shape[0])*100
print("The accuracy of Vader sentiment compared to airline  sentiments: " + str(accuracy))
```

    The accuracy of Vader sentiment compared to airline  sentiments: 54.65846994535519
    

## Q1:

### WordCloud - Negative Tweets

### Removing stop words and unimportant words


```python
# I came up with this list of custom words through an iterative process - Considered words which do not carry any information in this context
custom_stopwords_neg = list(stopwords.words('english'))
custom_stopwords_neg.extend(string.punctuation)
custom_stopwords_neg.extend(('americanair','usairways','united','southwestair','virginamerica','jetblue', 'http', 'co', 'i\'m', 'i', '\'m', 'amp', 'ca', 'wo', '\'s','n\'t', '\'ve', '\'\''))
```


```python
cloudstr = " ".join(tweet for tweet in data[data['polarity']=='negative']['text'])
wordcloud = WordCloud(background_color = "white", max_words=30, stopwords=custom_stopwords_neg).generate(cloudstr)

plt.figure(figsize=(18,12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off");
```


![png](output_39_0.png)


## Customers were mostly talking about the above topics like "Cancelled Flight", "Delay", "Customer Service", "time" and so on when they tweeted negatively. The most frequent words used can also be seen below.


```python
wordcloud.words_
```




    {'flight': 1.0,
     'Cancelled Flightled': 0.30813475760065734,
     'get': 0.2933442892358258,
     'time': 0.25554642563681185,
     'hour': 0.25472473294987674,
     'customer service': 0.2456861133935908,
     'plane': 0.24404272801972063,
     'bag': 0.21281840591618734,
     'delay': 0.20788824979457682,
     'one': 0.1972062448644207,
     'day': 0.17337715694330322,
     'us': 0.16926869350862778,
     'still': 0.1676253081347576,
     'delayed': 0.15940838126540674,
     'help': 0.1561216105176664,
     'airline': 0.15283483976992604,
     'today': 0.15283483976992604,
     'need': 0.14790468364831552,
     'seat': 0.14461791290057519,
     'Cancelled Flighted': 0.142974527526705,
     'gate': 0.14050944946589974,
     'phone': 0.13393590797041907,
     'airport': 0.1322925225965489,
     'flight Cancelled': 0.1265406737880033,
     "can't": 0.11832374691865243,
     'call': 0.11832374691865243,
     'service': 0.11668036154478226,
     'would': 0.11585866885784717,
     'agent': 0.11585866885784717,
     'thank': 0.114215283483977}



## But the single word frequency doesn't give much info about the topics. Let's print the most frequent bi-grams.

### Sentence tokenizing all tweets


```python
neg_sentences = []
neg_df = data[data['polarity']=='negative']
for i in range(0, neg_df.shape[0]):
    neg_sentences.extend(tweet for tweet in sent_tokenize(neg_df.iloc[i]['text']))
neg_sentences[:5]
```




    ['it\'s really aggressive to blast obnoxious "entertainment" in your guests\' faces &amp; they have little recourse',
     "and it's a really big bad thing about it",
     "seriously would pay $30 a flight for seats that didn't have this playing.",
     "it's really the only bad thing about flying VA",
     "Well, I didn't…but NOW I DO!"]



### Word tokenizing each of the sentecnes & extracting bi-grams


```python
bigram_neg = []
for line in neg_sentences:
    tokens = (word for word in word_tokenize(line) if word.lower() not in custom_stopwords_neg)
    bigram_neg.extend(ngrams(tokens, 2))
```

    C:\Users\HPHP\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: DeprecationWarning: generator 'ngrams' raised StopIteration
      after removing the cwd from sys.path.
    


```python
fdist_neg = nltk.FreqDist(bigram_neg)
fdist_neg.most_common(20)
```




    [(('Cancelled', 'Flightled'), 369),
     (('customer', 'service'), 269),
     (('Cancelled', 'Flighted'), 168),
     (('Late', 'Flight'), 124),
     (('flight', 'Cancelled'), 118),
     (('Cancelled', 'Flight'), 102),
     (('Booking', 'Problems'), 100),
     (('Flightled', 'flight'), 69),
     (('Late', 'Flightr'), 67),
     (('Flight', 'Booking'), 65),
     (('flight', 'delayed'), 60),
     (('2', 'hours'), 49),
     (('call', 'back'), 45),
     (('hour', 'delay'), 42),
     (('delayed', 'flight'), 36),
     (('Flighted', 'flight'), 36),
     (('trying', 'get'), 35),
     (('Flight', 'flight'), 35),
     (('3', 'hours'), 34),
     (('get', 'home'), 34)]



## We can see that above 2-word phrases occurred most frequently in "Negative tweets".

## Q2:

### WordCloud - Positive Tweets


```python
custom_stopwords_pos = list(stopwords.words('english'))
custom_stopwords_pos.extend(string.punctuation)
custom_stopwords_pos.extend(('americanair','usairways','united','southwestair','virginamerica','jetblue', 'http', 'co', 'i\'m', 'i', '\'m', 'amp', 'ca', 'wo', '\'s','n\'t', '\'ve', '\'\''))
```


```python
cloudstr = " ".join(tweet for tweet in data[data['polarity']=='positive']['text'])
wordcloud = WordCloud(background_color = "white", max_words=30, stopwords=custom_stopwords_pos).generate(cloudstr)

plt.figure(figsize=(18,12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off");
```


![png](output_52_0.png)


## Customers were mostly talking about the above topics like "Thank", "Flight", "Help" and so on when they tweeted positively. The most frequent words used can also be seen below.


```python
wordcloud.words_
```




    {'Thank': 1.0,
     'flight': 0.7254313578394599,
     'help': 0.3765941485371343,
     'get': 0.2955738934733683,
     'please': 0.2145536384096024,
     'time': 0.21305326331582897,
     'would': 0.16729182295573894,
     'bag': 0.16654163540885222,
     'need': 0.16504126031507876,
     'plane': 0.16054013503375844,
     'us': 0.15978994748687173,
     'one': 0.15603900975243812,
     'customer service': 0.15153788447111777,
     'great': 0.1485371342835709,
     'like': 0.1417854463615904,
     'day': 0.13953488372093023,
     'hour': 0.13878469617404351,
     'yes': 0.13728432108027006,
     'today': 0.13728432108027006,
     'know': 0.13278319579894973,
     'want': 0.13278319579894973,
     'ticket': 0.1312828207051763,
     'guy': 0.1312828207051763,
     'good': 0.13053263315828958,
     'got': 0.13053263315828958,
     'love': 0.12003000750187547,
     'seat': 0.11852963240810202,
     'airline': 0.1177794448612153,
     'still': 0.11702925731432859,
     'better': 0.11177794448612154}



## But the single word frequency doesn't give much info about the topics. Let's print the most frequent bi-grams.

### Sentence tokenizing positive tweets


```python
pos_sentences = []
pos_df = data[data['polarity']=='positive']
for i in range(0, pos_df.shape[0]):
    pos_sentences.extend(tweet for tweet in sent_tokenize(pos_df.iloc[i]['text']))
pos_sentences[:5]
```




    ['yes, nearly every time I fly VX this “ear worm” won’t go away :)',
     'Really missed a prime opportunity for Men Without Hats parody, there.',
     'it was amazing, and arrived an hour early.',
     "You're too good to me.",
     'I &lt;3 pretty graphics.']



### Word tokenizing each of the sentecnes & extracting bi-grams


```python
bigram_pos = []
for line in pos_sentences:
    tokens = (word for word in word_tokenize(line) if word.lower() not in custom_stopwords_pos)
    bigram_pos.extend(ngrams(tokens, 2))
```

    C:\Users\HPHP\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: DeprecationWarning: generator 'ngrams' raised StopIteration
      after removing the cwd from sys.path.
    


```python
fdist_pos = nltk.FreqDist(bigram_pos)
fdist_pos.most_common(20)
```




    [(('customer', 'service'), 192),
     (('Cancelled', 'Flightled'), 130),
     (('Late', 'Flight'), 65),
     (('Late', 'Flightr'), 52),
     (('flight', 'Cancelled'), 51),
     (('Cancelled', 'Flighted'), 51),
     (('Booking', 'Problems'), 42),
     (('Cancelled', 'Flight'), 40),
     (('please', 'help'), 37),
     (('would', 'like'), 36),
     (('2', 'hours'), 34),
     (('looks', 'like'), 34),
     (('need', 'help'), 34),
     (('get', 'home'), 32),
     (('Please', 'help'), 32),
     (('first', 'class'), 28),
     (('change', 'flight'), 26),
     (('flight', 'attendant'), 26),
     (('Flightled', 'flight'), 25),
     (('great', 'flight'), 25)]



## We can see that above 2-word phrases occurred most frequently in positive tweets.

# Q3

### How many tweets mention each of the airlines?


```python
pd.DataFrame(data.groupby('airline')['text'].count().sort_values(ascending=False)).plot(kind='bar', figsize=(18,8), title='Airlines Vs Tweets Count')
plt.ylabel('Tweets Count')
plt.xlabel('Airlines')
```




    Text(0.5,0,'Airlines')




![png](output_64_1.png)


#### United airlines was mentioned the most with 3822 number of tweets, whereas Virgin America was only referred 504 times.

### Which airline has most number of users tweeting about them?


```python
pd.DataFrame(data.groupby('airline')['name'].nunique().sort_values(ascending=False)).plot(kind='bar', figsize=(18,8), title='Airlines Vs Users Count', colormap='Greens_r')
plt.ylabel('Users Count')
plt.xlabel('Airlines')
```




    Text(0.5,0,'Airlines')




![png](output_67_1.png)


#### Again, the same pattern follows for number of users here. While United Airlines is at top with 1989 unique users tweeting about them, Virgin America is at the bottom with 504 users tweeting about them. However, the middle order is completely different from above pattern.

## In which hours do users tweets most?


```python
data['tweet_hour'] = pd.DatetimeIndex(data['tweet_created']).hour
data[['tweet_created','tweet_hour']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_created</th>
      <th>tweet_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2/24/2015 11:35</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2/24/2015 11:15</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/24/2015 11:15</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/24/2015 11:15</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2/24/2015 11:14</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(20,12))
sns.countplot(data['tweet_hour'], hue=data['airline']);
```


![png](output_71_0.png)


### From above plot, it's obvious that most tweets are posted between 9 AM - 3 PM for majority of the airlines.

# Q4:


```python
data_re = data[data['retweet_count']>0]
data_re.shape
```




    (767, 19)




```python
data_re.sentiment.value_counts().plot(kind='bar', title="Sentiment Among Tweets Which Are Re-Tweeted", figsize=(20,12))
plt.xlabel('Sentiments')
plt.ylabel('Tweets Count')
```




    Text(0,0.5,'Tweets Count')




![png](output_75_1.png)


### So most tweets, which have re-tweets to them are Negative followed by Very Negative.

# Q5:


```python
data_vp = data[data['sentiment']=='V.Positive']
```


```python
pd.DataFrame(data_vp.groupby('name')['name'].count().sort_values(ascending=False))[:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Logunov_Daniil</th>
      <td>10</td>
    </tr>
    <tr>
      <th>geekstiel</th>
      <td>9</td>
    </tr>
    <tr>
      <th>kbosspotter</th>
      <td>7</td>
    </tr>
    <tr>
      <th>georgetietjen</th>
      <td>7</td>
    </tr>
    <tr>
      <th>MeeestarCoke</th>
      <td>6</td>
    </tr>
    <tr>
      <th>NoviceFlyer</th>
      <td>6</td>
    </tr>
    <tr>
      <th>chagaga2013</th>
      <td>6</td>
    </tr>
    <tr>
      <th>NickTypesWords</th>
      <td>6</td>
    </tr>
    <tr>
      <th>flemmingerin</th>
      <td>5</td>
    </tr>
    <tr>
      <th>JetBlueNews</th>
      <td>5</td>
    </tr>
    <tr>
      <th>The_Playmaker20</th>
      <td>5</td>
    </tr>
    <tr>
      <th>kzone7</th>
      <td>5</td>
    </tr>
    <tr>
      <th>throthra</th>
      <td>5</td>
    </tr>
    <tr>
      <th>GREATNESSEOA</th>
      <td>5</td>
    </tr>
    <tr>
      <th>jasemccarty</th>
      <td>5</td>
    </tr>
    <tr>
      <th>DavidAlfieWard</th>
      <td>4</td>
    </tr>
    <tr>
      <th>D_WaYnE_01</th>
      <td>4</td>
    </tr>
    <tr>
      <th>lizStonewriter</th>
      <td>4</td>
    </tr>
    <tr>
      <th>danteusa1</th>
      <td>4</td>
    </tr>
    <tr>
      <th>logantracey</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



### Above users are the top 20 users, who most frequently tweets very positively.

# Q6:


```python
data_vn = data[data['sentiment']=='V.Negative']
```


```python
pd.DataFrame(data_vn.groupby('name')['name'].count().sort_values(ascending=False))[:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Evan_Flay</th>
      <td>7</td>
    </tr>
    <tr>
      <th>Aero0729</th>
      <td>7</td>
    </tr>
    <tr>
      <th>idk_but_youtube</th>
      <td>6</td>
    </tr>
    <tr>
      <th>weezerandburnie</th>
      <td>6</td>
    </tr>
    <tr>
      <th>otisday</th>
      <td>6</td>
    </tr>
    <tr>
      <th>lj_verde</th>
      <td>5</td>
    </tr>
    <tr>
      <th>kevinforgoogle</th>
      <td>5</td>
    </tr>
    <tr>
      <th>ezemanalyst</th>
      <td>5</td>
    </tr>
    <tr>
      <th>GREATNESSEOA</th>
      <td>5</td>
    </tr>
    <tr>
      <th>thomashoward88</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Allisonjones704</th>
      <td>5</td>
    </tr>
    <tr>
      <th>TinaHovsepian</th>
      <td>5</td>
    </tr>
    <tr>
      <th>riricesq</th>
      <td>5</td>
    </tr>
    <tr>
      <th>chagaga2013</th>
      <td>5</td>
    </tr>
    <tr>
      <th>throthra</th>
      <td>5</td>
    </tr>
    <tr>
      <th>navydocdro</th>
      <td>4</td>
    </tr>
    <tr>
      <th>aminghadersohi</th>
      <td>4</td>
    </tr>
    <tr>
      <th>ElmiraBudMan</th>
      <td>4</td>
    </tr>
    <tr>
      <th>TinaIsBack</th>
      <td>4</td>
    </tr>
    <tr>
      <th>flemmingerin</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



### Above users are the top 20 users, who most frequently tweets very negatively.

# Bonus Questions

## BQ1:


```python
np.mean(data.groupby('name')['airline'].nunique())
```




    1.0283080119465005



### On average, a customer tweets about 1.03 number of airlines out of these 6 airlines.

## BQ2: This question can not be answered with the scope of the available dataset, since the data is from a range of only 9 days.

## BQ3:

### There is no column in the data to indicate if a tweet is in fact a re-tweet. So, I will consider all the tweets containing the string 'RT' in them as re-tweets and all other tweets as direct tweets.


```python
" RT " in "Hello this TRT is test"
```




    False




```python
# Creating a dataframe with direct tweets count at user level
direct_tweets = data[~data['text'].str.contains(" RT ")][['name','text']].groupby('name').count()
direct_tweets.rename(columns={'text':'DTCount'},inplace=True)
direct_tweets.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DTCount</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0504Traveller</th>
      <td>5</td>
    </tr>
    <tr>
      <th>0veranalyser</th>
      <td>2</td>
    </tr>
    <tr>
      <th>0xjared</th>
      <td>1</td>
    </tr>
    <tr>
      <th>10Eshaa</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1234567890_</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Creating a dataframe with re-tweets count at user level
re_tweets = data[data['text'].str.contains(" RT ")][['name','text']].groupby('name').count()
re_tweets.rename(columns={'text':'RTCount'},inplace=True)
re_tweets.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RTCount</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AbeFroman</th>
      <td>1</td>
    </tr>
    <tr>
      <th>AdamJEstep</th>
      <td>1</td>
    </tr>
    <tr>
      <th>AngBeTweetin</th>
      <td>1</td>
    </tr>
    <tr>
      <th>AritheGenius</th>
      <td>1</td>
    </tr>
    <tr>
      <th>BALLarsen</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Using inner join to get users who both tweeted and re-tweeted
tweets_df = pd.merge(direct_tweets, re_tweets, left_index=True, right_index=True,)
tweets_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DTCount</th>
      <th>RTCount</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BALLarsen</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>MOCBlogger</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>RyanEthanBeard</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>TheFireTracker2</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>bostongarden</th>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>danihampton</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>goodenufmother</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>mitchsunderland</th>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>vincenzolandino</th>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>yourlocalnyer</th>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
tweets_df.shape
```




    (10, 2)



### There are only 10 users who posted at least 1 tweet and 1 re-tweet


```python
tweets_df.sort_values(by=['DTCount','RTCount'], ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DTCount</th>
      <th>RTCount</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bostongarden</th>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>vincenzolandino</th>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>mitchsunderland</th>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>goodenufmother</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>RyanEthanBeard</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>yourlocalnyer</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>BALLarsen</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>MOCBlogger</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>TheFireTracker2</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>danihampton</th>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### So, above are the users who most frequently tweeted & re-tweeted from top to bottom.
