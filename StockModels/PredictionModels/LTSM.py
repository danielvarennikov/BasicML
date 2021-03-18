import math
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from numpy.random import seed
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from tensorflow.python.framework.random_seed import set_random_seed


class LTSM:
    lstm_units = 50
    dropout_prob = 0.9
    optimizer = 'adam'
    epochs = 1
    batch_size = 1

    model_seed = 100

    # Set seeds to ensure same output results
    seed(101)

    set_random_seed(model_seed)

    def get_x_y(self,data, N, offset):
        """
        Split data into x (features) and y (target)
        """
        x, y = [], []
        for i in range(offset, len(data)):
            x.append(data[i - N:i])
            y.append(data[i])
        x = np.array(x)
        y = np.array(y)

        return x, y

    def get_x_scaled_y(self, data, N, offset):
        """
        Split data into x (features) and y (target)
        We scale x to have mean 0 and std dev 1, and return this.
        We do not scale y here.
        Inputs
            data     : pandas series to extract x and y
            N
            offset
        Outputs
            x_scaled : features used to predict y. Scaled such that each element has mean 0 and std dev 1
            y        : target values. Not scaled
            mu_list  : list of the means. Same length as x_scaled and y
            std_list : list of the std devs. Same length as x_scaled and y
        """
        x_scaled, y, mu_list, std_list = [], [], [], []
        for i in range(offset, len(data)):
            mu_list.append(np.mean(data[i - N:i]))
            std_list.append(np.std(data[i - N:i]))
            x_scaled.append((data[i - N:i] - mu_list[i - offset]) / std_list[i - offset])
            y.append(data[i])
        x_scaled = np.array(x_scaled)
        y = np.array(y)

        return x_scaled, y, mu_list, std_list

    def train_pred_eval_model(self, x_train_scaled, \
                              y_train_scaled, \
                              x_cv_scaled, \
                              y_cv, \
                              mu_cv_list, \
                              std_cv_list, \
                              lstm_units=50, \
                              dropout_prob=0.5, \
                              optimizer='adam', \
                              epochs=1, \
                              batch_size=1):
        '''
        Train model, do prediction, scale back to original range and do evaluation
        Use LSTM here.
        Returns rmse, mape and predicted values
        Inputs
            x_train_scaled  : e.g. x_train_scaled.shape=(451, 9, 1). Here we are using the past 9 values to predict the next value
            y_train_scaled  : e.g. y_train_scaled.shape=(451, 1)
            x_cv_scaled     : use this to do predictions
            y_cv            : actual value of the predictions
            mu_cv_list      : list of the means. Same length as x_scaled and y
            std_cv_list     : list of the std devs. Same length as x_scaled and y
            lstm_units      : lstm param
            dropout_prob    : lstm param
            optimizer       : lstm param
            epochs          : lstm param
            batch_size      : lstm param
        Outputs
            rmse            : root mean square error
            mape            : mean absolute percentage error
            est             : predictions
        '''
        # Create the LSTM network
        model = Sequential()
        model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(x_train_scaled.shape[1], 1)))
        model.add(Dropout(dropout_prob))  # Add dropout with a probability of 0.5
        model.add(LSTM(units=lstm_units))
        model.add(Dropout(dropout_prob))  # Add dropout with a probability of 0.5
        model.add(Dense(1))

        # Compile and fit the LSTM network
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        model.fit(x_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

        # Do prediction
        est_scaled = model.predict(x_cv_scaled)
        est = (est_scaled * np.array(std_cv_list).reshape(-1, 1)) + np.array(mu_cv_list).reshape(-1, 1)

        # Calculate RMSE and MAPE
        #     print("x_cv_scaled = " + str(x_cv_scaled))
        #     print("est_scaled = " + str(est_scaled))
        #     print("est = " + str(est))
        rmse = math.sqrt(mean_squared_error(y_cv, est))
        mape = 1  # get_mape(y_cv, est)

        return rmse, mape, est

    # Converting dataset into x_train and y_train
    def get_xy_train(self, debug, train, N):
        # Here we only scale the train dataset, and not the entire dataset to prevent information leak
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(np.array(train['adj_close']).reshape(-1, 1))
        if debug:
            print("scaler.mean_ = " + str(scaler.mean_))
            print("scaler.var_ = " + str(scaler.var_))

        # Split into x and y
        x_train_scaled, y_train_scaled = self.get_x_y(train_scaled, N, N)
        if debug:
            print("x_train_scaled.shape = " + str(x_train_scaled.shape))  # (446, 7, 1)
            print("y_train_scaled.shape = " + str(y_train_scaled.shape))  # (446, 1)
        return x_train_scaled, y_train_scaled

    # Scale the dataset
    def get_scaled_dataset(self, train_cv, N, num_train, debug):
        # Split into x and y
        x_cv_scaled, y_cv, mu_cv_list, std_cv_list = self.get_x_scaled_y(np.array(train_cv['adj_close']).reshape(-1, 1), N,
                                                                    num_train)
        if debug:
            print("x_cv_scaled.shape = " + str(x_cv_scaled.shape))
            print("y_cv.shape = " + str(y_cv.shape))
            print("len(mu_cv_list) = " + str(len(mu_cv_list)))
            print("len(std_cv_list) = " + str(len(std_cv_list)))

        scaler_final = StandardScaler()
        train_cv_scaled_final = scaler_final.fit_transform(np.array(train_cv['adj_close']).reshape(-1, 1))
        if debug:
            print("scaler_final.mean_ = " + str(scaler_final.mean_))
            print("scaler_final.var_ = " + str(scaler_final.var_))
        return x_cv_scaled, y_cv, mu_cv_list, std_cv_list

    # Create the LSTM network
    def create_network(self, x_train_scaled, y_train_scaled):
        model = Sequential()
        model.add(LSTM(units=self.lstm_units, return_sequences=True, input_shape=(x_train_scaled.shape[1], 1)))
        model.add(Dropout(self.dropout_prob))  # Add dropout with a probability of 0.5
        model.add(LSTM(units=self.lstm_units))
        model.add(Dropout(self.dropout_prob))  # Add dropout with a probability of 0.5
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        model.fit(x_train_scaled, y_train_scaled, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        return model

    # Do prediction
    def predict(self, model, x_cv_scaled, std_cv_list, mu_cv_list, y_cv):
        est_scaled = model.predict(x_cv_scaled)
        est = (est_scaled * np.array(std_cv_list).reshape(-1, 1)) + np.array(mu_cv_list).reshape(-1, 1)
        print("est.shape = " + str(est.shape))

        # Calculate RMSE
        rmse_bef_tuning = math.sqrt(mean_squared_error(y_cv, est))
        print("RMSE = %0.3f" % rmse_bef_tuning)

        # Calculate MAPE
        mape_pct_bef_tuning = 1  # get_mape(y_cv, est)
        print("MAPE = %0.3f%%" % mape_pct_bef_tuning)
        return est

    # plot
    def draw_graph(self, est, y_cv, train, cv, test):
        # Plot adjusted close over time
        rcParams['figure.figsize'] = 10, 8  # width 10, height 8

        est_df = pd.DataFrame({'est': est.reshape(-1),
                               'y_cv': y_cv.reshape(-1),
                               'date': cv['date']})

        ax = train.plot(x='date', y='adj_close', style='b-', grid=True)
        ax = cv.plot(x='date', y='adj_close', style='y-', grid=True, ax=ax)
        ax = test.plot(x='date', y='adj_close', style='g-', grid=True, ax=ax)
        ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
        ax.legend(['train', 'dev', 'test', 'est'])
        ax.set_xlabel("date")
        ax.set_ylabel("USD")
        plt.show()

    def LTSM_runner(self, debug, train, test, cv, N, train_cv, num_train):
        x_train_scaled, y_train_scaled = self.get_xy_train(debug, train, N)
        x_cv_scaled, y_cv, mu_cv_list, std_cv_list = self.get_scaled_dataset(train_cv, N, num_train, debug)
        model = self.create_network(x_train_scaled, y_train_scaled)
        est = self.predict(model, x_cv_scaled, std_cv_list, mu_cv_list, y_cv)
        self.draw_graph(est, y_cv, train, cv, test)

