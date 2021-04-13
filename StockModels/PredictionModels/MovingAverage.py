import math
import numpy as np
from sklearn.metrics import mean_squared_error


# Given a dataframe, get prediction at timestep t using values from t-1, t-2, ..., t-N.
# Using simple moving average.
# Inputs
#    df         : dataframe with the values you want to predict. Can be of any length.
#    target_col : name of the column you want to predict e.g. 'adj_close'
#    N          : get prediction at timestep t using values from t-1, t-2, ..., t-N
#    pred_min   : all predictions should be >= pred_min
#    offset     : for df we only do predictions for df[offset:]. e.g. offset can be size of training set
# Outputs
#    pred_list  : list. The predictions for target_col. np.array of length len(df)-offset.


class MovingAverage:

    # Compute mean absolute percentage error (MAPE)
    def get_mape(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def get_single_mov_avg(self, df, target_col, N, pred_min, offset):

        # Get the mean value of the previous N values in that column
        pred_list = df[target_col].rolling(window=N, min_periods=1).mean()

        # Add one timestep to the predictions
        pred_list = np.concatenate((np.array([np.nan]), np.array(pred_list[:-1])))

        # If the values are < pred_min, set it to be pred_min (e.g. no need for negative values)
        pred_list = np.array(pred_list)
        pred_list[pred_list < pred_min] = pred_min

        return pred_list[offset:]

    def calculate_moving_average(self, debug, train_cv, Nmax, num_train, cv):
        # Predict using moving averages
        RMSE = []
        mape = []

        # N is no. of samples to use to predict the next value
        for N in range(1, Nmax + 1):

            # The current prediction e.g. t - 2
            est_list = self.get_single_mov_avg(train_cv, 'adj_close', N, 0, num_train)
            cv.loc[:, 'est' + '_N' + str(N)] = est_list
            RMSE.append(math.sqrt(mean_squared_error(est_list, cv['adj_close'])))
            mape.append(self.get_mape(cv['adj_close'], est_list))
        if debug:
            print('Mean squared error = ' + str(RMSE))
            print('Mean absolute percentage error = ' + str(mape))
        return RMSE, mape

    def MovingAverage_runner(self, debug, cv, N, train_cv, num_train):

        return self.calculate_moving_average(debug, train_cv, N, num_train, cv)
