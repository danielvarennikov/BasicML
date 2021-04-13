import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


# DecisionTreeRegressor
class DTRSVR:

    def DTRSVR_runner(self, mode, df):
        
        prices = df[df.columns[0:2]]
        prices["timestamp"] = pd.to_datetime(prices.date).astype(int) // (10 ** 9)
        dates = prices['date']
        prices = prices.drop(['date'], axis=1)

        dataset = prices.values
        X = dataset[:, 1].reshape(-1, 1)
        Y = dataset[:, 0:1]

        validation_size = 0.15
        seed = 7
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size,
                                                                        random_state=seed)

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
        plt.xlabel("Date", fontsize='large')
        plt.ylabel("Price", fontsize='large')
        plt.plot(dates, Y)
        plt.plot(dates, predictions)
        plt.legend(["Original", "Predictions"])

        plt.show()
