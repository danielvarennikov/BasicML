import numpy as np

from StockModels.DataFrame import DataFrame
from StockModels.Drawer import Drawer
from StockModels.PredictionModels.LTSM import LTSM
from StockModels.PredictionModels.DTRSVR import DTRSVR
from StockModels.PredictionModels.MovingAverage import MovingAverage
from StockModels.PredictionModels.LinearRegressionClass import LinearRegressionClass

np.warnings.filterwarnings('ignore')


def parse_stock(current_stock):
    if int(current_stock) == 1:
        path = "./StockModels/StockDatasets/GOOGL.csv"
    elif int(current_stock) == 2:
        path = "./StockModels/StockDatasets/AMD.csv"
    elif int(current_stock) == 3:
        path = "./StockModels/StockDatasets/VTI.csv"

    return path


# --------------Training params------------#

Nmax = 9
debug = False
pred_algo = -1
path = ""


# ---------------Start:-------------------#

def main():
    dataframe_ref = DataFrame()
    drawer_ref = Drawer()
    ltsm_ref = LTSM()
    dtr_ref = DTRSVR()

    # Begin interaction with the user
    while True:

        # Get the stock and the prediction model from the user
        current_stock = drawer_ref.prompt_stock()
        pred_algo = drawer_ref.prompt_model()
        path = parse_stock(current_stock)

        # Get the formatted dataframe
        df = dataframe_ref.read_data(path)

        # Get sizes
        num_cv, num_test, num_train = dataframe_ref.get_sizes(debug, df)

        # Get teh corresponding parts of the dataset
        train, cv, train_cv, test = dataframe_ref.split_df(debug, df, num_cv, num_train)

        # Graph the slices
        if debug:
            drawer_ref.graph_split(train, cv, test)

        if int(pred_algo) == 1:
            current_model = MovingAverage()
            RMSE, mape = current_model.MovingAverage_runner(debug, cv, Nmax, train_cv, num_train)
            drawer_ref.graph_result(train, cv, test, Nmax, RMSE, mape)
        elif int(pred_algo) == 2:
            current_model = LinearRegressionClass()
            current_model.LinearRegression_runner(debug,cv, Nmax, train_cv, num_train)
            drawer_ref.graph_result(train, cv, test, Nmax)
        elif int(pred_algo) == 3:
            ltsm_ref.LTSM_runner(debug, train, test, cv, Nmax, train_cv, num_train)
        elif int(pred_algo) == 4:
            dtr_ref.DTRSVR_runner(0, df)
        elif int(pred_algo) == 5:
            dtr_ref.DTRSVR_runner(1, df)


if __name__ == "__main__":
    main()
