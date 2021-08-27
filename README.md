# Useful code snippets
This is a collection of useful code snippets for different classes.
## Getting Data
### Getting data from a website
```
import requests

url = 'https://www.worldometers.info/coronavirus/'
response = requests.get(url)

# Create a BeautifulSoup object
soup = BeautifulSoup(response.text, "html.parser")

# Print the BeautifulSoup object
print(soup)
```
### Getting financial data
```
import pandas as pd
import pandas_datareader as web
import datetime as dt

start = dt.datetime(2017, 1, 1)
end = dt.datetime(2020, 8, 4)

df_stock = web.DataReader('TSLA', 'yahoo', start, end)
```

### Getting news data
```
from newsapi.newsapi_client import NewsApiClient
import datetime

newsapi = NewsApiClient(api_key='YOUR_API_KEY')

all_articles = newsapi.get_everything(q='Elon Musk',
                                      from_param=datetime.date(2020, 8, 4),
                                      to=datetime.date(2020, 8, 4),
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)
```
### getting twitter data
```
import tweepy
import time

#Twitter API credentials
consumer_key = "YOUR CONSUMER KEY"
consumer_secret = "YOUR CONSUMER SECRET"
access_key = "YOUR ACCESS KEY"
access_secret = "YOUR ACCESS SECRET"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

tweets = []

#Define a for-loop to create a list of tweets
for tweet in tweepy.Cursor(api.search, q='Elon Musk', since='2020-08-04', until='2020-08-05', lang='en').items():
    tweets.append(tweet)

#Define a pandas DataFrame to store the date:
data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

#Display the first 10 elements of the dataframe:
display(data.head(10))
```

## Plotting
### Plotting with matplotlib
```
import matplotlib.pyplot as plt

plt.plot(df_stock['High'], 'r')
plt.plot(df_stock['Low'], 'b')
plt.legend(['High', 'Low'])
plt.show()
```

### Plotting with plotly
```
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['High'], name="High",
                         line_color='deepskyblue'))
fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['Low'], name="Low",
                         line_color='dimgray'))
fig.update_layout(title_text='Tesla Stock Price Data',
                  xaxis_rangeslider_visible=True)
fig.show()
```

### Plotting with seaborn
```
import seaborn as sns


sns.lineplot(x=df_stock.index, y=df_stock['High'])
sns.lineplot(x=df_stock.index, y=df_stock['Low'])
```

## Plotting with bokeh
```
from bokeh.plotting import figure, output_file, show
from bokeh.models import DatetimeTickFormatter

p = figure(plot_width=800, plot_height=350, x_axis_type='datetime')
p.line(df_stock.index, df_stock['High'], color='navy', alpha=0.5)
p.line(df_stock.index, df_stock['Low'], color='firebrick', alpha=0.5)
p.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d %B %Y"],
        days=["%d %B %Y"],
        months=["%d %B %Y"],
        years=["%d %B %Y"],
    )
show(p)
```
## Data Cleaning
### Removing punctuation
```
import re

text = "This &is [an] example? {of} string. with.? punctuation!!!!" # Sample string

# Remove punctuation
text = ''.join([c for c in text if c not in string.punctuation])
```

### Tokenization
```
import nltk

nltk.download('punkt')

text = "This is an example of nltk tokenization."

# Tokenize the text
words = nltk.word_tokenize(text)
```

### Removing stopwords
```
import nltk

nltk.download('stopwords')

text = "This is an example of nltk stopwords."

# Tokenize the text
words = nltk.word_tokenize(text)

# Remove stopwords
words = [word for word in words if word not in stopwords.words('english')]
```

### Stemming
```
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import PorterStemmer

text = "This is an example of nltk stemming."

# Tokenize the text
words = nltk.word_tokenize(text)

# Stem the words
ps = PorterStemmer()
words = [ps.stem(word) for word in words]
```

### Lemmatization
```
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

text = "This is an example of nltk lemmatization."

# Tokenize the text
words = nltk.word_tokenize(text)

# Stem the words
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word) for word in words]
```

## Feature Engineering
### Bag of Words
```
from sklearn.feature_extraction.text import CountVectorizer

text = ["This is an example of bag of words."]

# Create the bag of words
vectorizer = CountVectorizer()
vectorizer.fit(text)
bag_of_words = vectorizer.transform(text)

# Print the bag of words
print(bag_of_words)
```

### TF-IDF
```
from sklearn.feature_extraction.text import TfidfVectorizer

text = ["This is an example of tf-idf."]

# Create the tf-idf
vectorizer = TfidfVectorizer()
vectorizer.fit(text)
tfidf = vectorizer.transform(text)

# Print the tf-idf
print(tfidf)
```
## Web scraping
### Scraping a webpage
```
import scrapy

class BrickSetSpider(scrapy.Spider):
    name = "brickset_spider"
    start_urls = ['http://brickset.com/sets/year-2016']

    def parse(self, response):
        SET_SELECTOR = '.set'
        for brickset in response.css(SET_SELECTOR):
            NAME_SELECTOR = 'h1 ::text'
            yield {
                'name': brickset.css(NAME_SELECTOR).extract_first(),
            }
```


### Scraping a PDF
```
from io import BytesIO
from urllib.request import urlopen
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = BytesIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = urlopen(path)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text
```
