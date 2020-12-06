import numpy as np
import pprint
from hw2 import plot_results
#use output files of values and steps to re-plot all plots
files = [
    ('0.01-0.5-x.npy','0.01-0.5-y.npy'),
    ('0.01-0.7-x.npy','0.01-0.7-y.npy'),
    ('0.01-0.9-x.npy','0.01-0.9-y.npy'),
    ('0.02-0.5-x.npy','0.02-0.5-y.npy'),
    ('0.02-0.7-x.npy','0.02-0.7-y.npy'),
    ('0.02-0.9-x.npy','0.02-0.9-y.npy'),
    ('0.03-0.5-x.npy','0.03-0.5-y.npy'),
    ('0.03-0.7-x.npy','0.03-0.7-y.npy'),
    ('0.03-0.9-x.npy','0.03-0.9-y.npy')
    ]
values = {}
for fn in files:
    fx,fy = fn
    try:
        with open(fx,'rb') as f:
            x = np.load(f)
        with open(fy,'rb') as f:
            y = np.load(f)
    except:
        print(f'Failed to open one of: {fx}, {fy} !!!')
    a,l = fx[:-6].split('-')
    values[f'$\\alpha=${a},$\\lambda=${l}'] = ({'x':x,'y':y},float(fx.split('-')[0]),float(fx.split('-')[1]))

plot_results(values)