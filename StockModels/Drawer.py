from matplotlib import pyplot as plt
from pylab import rcParams


class Drawer:

    def prompt_stock(self):
        print("Hello and welcome to the stock predictor!")
        print("Please choose the desired stock:")
        print("1) Google")
        print("2) AMD")
        print("3) VTI")
        return input()

    def prompt_model(self):
        print("Please choose a prediction algorithm:")
        print("1) Moving average")
        print("2) Linear Regression")
        print("3) LTSM")
        print("4) Decision Tree Regressor")
        print("5) SVR")
        return input()

    # Show the splitting of the dataframe in a graphical manner
    def graph_split(self, train, cv, test):
        # Plot adjusted close over time
        rcParams['figure.figsize'] = 10, 8  # width 10, height 8

        ax = train.plot(x='date', y='adj_close', style='b-', grid=True)
        ax = cv.plot(x='date', y='adj_close', style='y-', grid=True, ax=ax)
        ax = test.plot(x='date', y='adj_close', style='g-', grid=True, ax=ax)
        ax.legend(['train', 'validation', 'test'])
        ax.set_xlabel("date")
        ax.set_ylabel("USD")
        plt.title('Dataframe split')
        plt.show()

    def graph_result(self, train, cv, test, Nmax):
        NUM_COLORS = Nmax + 4
        cm = plt.get_cmap('gist_rainbow')
        rcParams['figure.figsize'] = 10, 8  # width 10, height 8

        ax = train.plot(x='date', y='adj_close', color=cm(0 // 3 * 3.0 / NUM_COLORS), grid=True)
        ax = cv.plot(x='date', y='adj_close', color=cm(1 // 3 * 3.0 / NUM_COLORS), grid=True, ax=ax)
        ax = test.plot(x='date', y='adj_close', color=cm(2 // 3 * 3.0 / NUM_COLORS), grid=True, ax=ax)
        legend = ['train', 'validation', 'test']
        for N in range(1, Nmax + 1):
            legend.append('predictions with N=' + str(N))
            ax = cv.plot(x='date', y='est_N' + str(N), color=cm((N + 2) // 2 * 3.0 / NUM_COLORS), grid=True, ax=ax)

        ax.legend(legend)
        ax.set_xlabel("date")
        ax.set_ylabel("USD")
        plt.show()
