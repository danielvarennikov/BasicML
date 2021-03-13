from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# create a random dataset
def create_dataset(amount, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(amount):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val -=step

    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


# m -> y = mx + b
def get_fit_slope(x_coords, y_coords):
    return (mean(x_coords) * mean(y_coords) - mean(x_coords * y_coords)) / ((mean(x_coords) ** 2) - mean(x_coords ** 2))


# b -> y = mx + b
def get_fit_intercept(x_coords, y_coords, m):
    return (mean(y_coords) - (m * mean(x_coords)))


def get_regression_line(x_coords, m, b):
    return [(m * x) + b for x in x_coords]


def squared_error(ys_original, ys_line):
    return sum((ys_line - ys_original)**2)


# r^2 -> r^2 = 1 - SEy^ / SEy-
def coefficient_of_determination(ys_original, ys_line):
    y_mean_line = [mean(ys_original) for y in ys_original]
    squared_error_regr = squared_error(ys_original, ys_line)
    squared_error_y_mean = squared_error(ys_original, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

#xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
#ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

xs, ys = create_dataset(40 , 10, 2, correlation='pos')

m = get_fit_slope(xs, ys)
b = get_fit_intercept(xs, ys, m)
line = get_regression_line(xs, m, b)


# the error
r_squared = coefficient_of_determination(ys, line)
print(r_squared)

plt.scatter(xs, ys)
plt.plot(xs, line)
plt.show()
