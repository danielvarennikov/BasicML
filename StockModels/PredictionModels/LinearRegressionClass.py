import math
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


class LinearRegressionClass:

    # Compute mean absolute percentage error (MAPE)
    def get_mape(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def get_preds_lin_reg(self, df, target_col, N, pred_min, offset):

        # Create linear regression object
        regr = LinearRegression(fit_intercept=True)

        pred_list = []

        for i in range(offset, len(df['adj_close'])):
            X_train = np.array(range(len(df['adj_close'][i - N:i])))  # e.g. [0 1 2 3 4]
            y_train = np.array(df['adj_close'][i - N:i])  # e.g. [2944 3088 3226 3335 3436]
            X_train = X_train.reshape(-1, 1)  # e.g X_train =
            y_train = y_train.reshape(-1, 1)

            # Train the model
            regr.fit(X_train, y_train)
            pred = regr.predict(np.array(N).reshape(1, -1))

            pred_list.append(pred[0][0])  # Predict the footfall using the model

        # If the values are < pred_min, set it to be pred_min (e.g. no need for negative values)
        pred_list = np.array(pred_list)
        pred_list[pred_list < pred_min] = pred_min

        return pred_list

    def calculate_linear_regression(self, debug, train_cv, Nmax, num_train, cv):
        RMSE = []
        R2 = []
        mape = []

        # N is no. of samples to use to predict the next value
        for N in range(1, Nmax + 1):
            est_list = self.get_preds_lin_reg(train_cv, 'adj_close', N, 0, num_train)

            cv.loc[:, 'est' + '_N' + str(N)] = est_list
            RMSE.append(math.sqrt(mean_squared_error(est_list, cv['adj_close'])))
            R2.append(r2_score(cv['adj_close'], est_list))
            mape.append(self.get_mape(cv['adj_close'], est_list))
        if debug:
            print('RMSE = ' + str(RMSE))
            print('R2 = ' + str(R2))
            print('MAPE = ' + str(mape))

    def LinearRegression_runner(self, debug, cv, N, train_cv, num_train):
        self.calculate_linear_regression(debug, train_cv, N, num_train, cv)
