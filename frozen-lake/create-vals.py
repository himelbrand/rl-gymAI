import numpy as np
import pprint
from main import plot_results
files = [('0.01-0.7-x.npy','0.01-0.7-y.npy'),('0.01-0.9-x.npy','0.01-0.9-y.npy'),('0.02-0.7-x.npy','0.02-0.7-y.npy'),('0.02-0.9-x.npy','0.02-0.9-y.npy')]
values = {}
for fn in files:
    fx,fy = fn
    with open(fx,'rb') as f:
        x = np.load(f)
    with open(fy,'rb') as f:
        y = np.load(f)
    values[fx[:-6]] = ({'x':x,'y':y},float(fx.split('-')[0]),float(fx.split('-')[1]))

plot_results(values)