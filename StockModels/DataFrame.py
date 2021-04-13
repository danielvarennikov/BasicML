import pandas as pd


class DataFrame:

    stk_path = ""

    # proportion of dataset to be used as test set
    test_size = 0.2

    # proportion of dataset to be used as cross-validation set
    cv_size = 0.2

    def read_data(self, path):
        # Read the data
        df = pd.read_csv(path, sep=",")

        # Convert Date column to datetime
        df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

        # Change all column headings to be lower case, and remove spacing
        df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

        # Get month of each sample
        df['month'] = df['date'].dt.month

        # Sort by datetime
        df.sort_values(by='date', inplace=True, ascending=True)

        return df

    # Get sizes of each of the datasets
    def get_sizes(self, debug, dataframe, cv_size=cv_size, test_size=test_size):
        num_cv = int(cv_size * len(dataframe))
        num_test = int(test_size * len(dataframe))
        num_train = len(dataframe) - num_cv - num_test
        if debug:
            print("num_train = " + str(num_train))
            print("num_cv = " + str(num_cv))
            print("num_test = " + str(num_test))
        return num_cv, num_test, num_train

    # Split into train, cv, and test according to the sizes we got
    def split_df(self, debug, dataframe, num_cv, num_train):
        train = dataframe[:num_train]
        cv = dataframe[num_train:num_train + num_cv]
        train_cv = dataframe[:num_train + num_cv]
        test = dataframe[num_train + num_cv:]

        if debug:
            print("train.shape = " + str(train.shape))
            print("cv.shape = " + str(cv.shape))
            print("train_cv.shape = " + str(train_cv.shape))
            print("test.shape = " + str(test.shape))
        return train, cv, train_cv, test
