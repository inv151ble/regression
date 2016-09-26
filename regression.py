from __future__ import division
import pandas as pd
import random
import numpy
import os
from matplotlib import pyplot as plt

def func(x):
    return 179*x+500

def add_noise(x, value):
    return x*(1+random.uniform(-1, 1)*value)

def make_results(n, func, noise_v=0):
    return [(x, add_noise(func(x), noise_v)) for x in range(n)]

def standard_deviation(x_l):
    return (sum(map(lambda x: x**2, x_l))/len(x_l)-(numpy.mean(x_l))**2)**0.5

def build_regression(data):
    x_l, y_l = zip(*data)
    x_mean = numpy.mean(x_l)
    y_mean = numpy.mean(y_l)
    xy_mean = numpy.mean([x*y for x, y in data])
    x_sd = standard_deviation(x_l)
    y_sd = standard_deviation(y_l)
    correlation_coef = (xy_mean - x_mean*y_mean)/(x_sd*y_sd)
    print("x_mean: %d, y_mean: %d, xy_mean: %d, x_sd: %d, y_sd: %d" %
          (x_mean,y_mean,xy_mean,x_sd,y_sd))
    print("Correlation coefficient: %f" %correlation_coef)
    return lambda x: ((correlation_coef*(x-x_mean)*y_sd)/x_sd)+y_mean

def regression_error(y_regression, y_l):
    error = sum([numpy.abs((y-y_r)/y)*100 for y_r, y in zip(y_regression, y_l)])
    return error/len(y_l)

###Menu
menu = ''
directory = os.getcwd()
while (menu!='1' and menu!='2'):
    menu=raw_input('Press 1 to generate dataset, press 2 to read from file: ')
if menu == '2':
    df = pd.read_csv(directory+'/regr.csv')
    results = map(tuple, df.values.tolist())
else:
    menu_save = ''
    x_num=input('Enter number of generated points: ')
    noise_v=input('Enter noise value applied to function:')
    results = make_results(x_num, func, noise_v)
    df=pd.DataFrame(data=results)
    while (menu_save!='y' and menu_save!='n'):
        menu_save = raw_input('Save new dataset to file(y/n)? ')
    if menu_save == 'y': df.to_csv('regr.csv', index=False)
###

x_l, y_l = zip(*results)
regression = map(build_regression(results), x_l)
error = regression_error(regression, y_l)
plt.plot(x_l, y_l, 'b.', x_l, regression, 'r')
plt.title('Error=%f%%' %error)
plt.show()
