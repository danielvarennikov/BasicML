import pandas as pd
import quandl as qd
import math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')


def draw_graph():
    df['Close'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


# get the dataframe
df = qd.get("BCHARTS/COINFALCONEUR")

# the dataframe columms
df = df[['Open', 'High', 'Low', 'Close', 'Volume (BTC)']]
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100.0

# daily move
df['PCT_CHANGE'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# features
df = df[['Close', 'HL_PCT', 'PCT_CHANGE', 'Volume (BTC)']]

forecast_col = 'Close'
train_again = False

# fill non existent data
df.fillna(-99999, inplace=True)

# how many days into the fuure
forecast_out = int(math.ceil(0.01 * len(df)))
print("Days into the future: " + str(forecast_out))
# shift the column 'into the future'
df['label'] = df[forecast_col].shift(-forecast_out)

# features: x, labels: y
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
# the predicted values
X_lately = X[-forecast_out:]
# up to 90 percent
X = X[:-forecast_out]


df.dropna(inplace=True)
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

if train_again:
    print("Training....")

    # prediction algorithm -> use all cores
    clf = LinearRegression(n_jobs=-1)

    # train the model
    clf.fit(X_train, Y_train)

    # save the classifier
    with open('linearregression.pickle', 'wb') as f:
        pickle.dump(clf, f)

else:
    # get the classifier from the file
    pickle_in = open('linearregression.pickle', 'rb')
    print("Reading old data....")
    clf = pickle.load(pickle_in)

# check the results
accuracy = clf.score(X_test, Y_test)
print("Accuracy: " + str(accuracy))
forecast_set = clf.predict(X_lately)
print(forecast_set, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

draw_graph()
