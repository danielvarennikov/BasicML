import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR



# DecisionTreeRegressor
class DTRSVR:

    def DTRSVR_runner(self, mode, df):
        prices = df[df.columns[0:2]]

        prices["timestamp"] = pd.to_datetime(prices.date).astype(int) // (10 ** 9)
        prices = prices.drop(['date'], axis=1)

        dataset = prices.values
        X = dataset[:, 1].reshape(-1, 1)
        Y = dataset[:, 0:1]

        validation_size = 0.15
        seed = 7
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size,
                                                                        random_state=seed)

        # Future prediction, add dates here for which you want to predict
        dates = ["2020-12-10", "2020-12-11", "2020-12-12", "2020-12-13", "2020-12-14", ]
        # convert to time stamp
        for dt in dates:
            datetime_object = datetime.strptime(dt, "%Y-%m-%d")
            timestamp = datetime.timestamp(datetime_object)
            # to array X
            np.append(X, int(timestamp))

        # Define model
        if mode == 0:
            model = DecisionTreeRegressor()
        else:
            model = SVR()

        # Fit to model
        model.fit(X_train, Y_train)
        # predict
        predictions = model.predict(X)

        fig = plt.figure(figsize=(24, 12))

        plt.plot(X, Y)
        plt.plot(X, predictions)
        plt.legend(["Original", "Predictions"])
        plt.show()
