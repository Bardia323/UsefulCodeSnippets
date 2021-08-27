# UsefulCodeSnippets
This is a collection of useful code snippets for different classes.
# Useful code snippets
## Getting Data
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

