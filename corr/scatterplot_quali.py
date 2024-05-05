import pandas as pd
import numpy as np
from numpy.random import uniform
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union


def x_noise_hue(data:pd.DataFrame, x:str, y:str, hue:str, dist:float):
    ptable = pd.pivot_table(data=data.reset_index(), values='index', index=(x, y), columns=hue, aggfunc='count')
    row_sums = ptable.sum(axis=1)
    for col in ptable.columns:
        ptable[col] = 2*dist*ptable[col]/row_sums
    ptable_max = ptable.cumsum(axis=1) - dist
    ptable_min = ptable_max.shift(periods=1, axis=1).fillna(-dist)
    ptable_max, ptable_min = ptable_max.stack(), ptable_min.stack()
    ptable_max.name, ptable_min.name = 'max', 'min'
    data = pd.merge(data, ptable_min, left_on=(x, y, hue), right_index=True)
    data = pd.merge(data, ptable_max, left_on=(x, y, hue), right_index=True)
    data['diff'] = data['max'] - data['min']
    data['rand'] = np.random.uniform(0, 1, size=len(data))
    return data['rand']*data['diff'] + data['min']


def scatterplot_quali_(data:pd.DataFrame, x:str, y:str, hue:str=None, ax=None, **kwargs):
    dist = 0.4
    unique_x, unique_y = data[x].cat.categories, data[y].cat.categories
    map_x, map_y = {val: i for (i, val) in enumerate(unique_x)}, {val: i for (i, val) in enumerate(unique_y)}
    x_, y_ = data[x].map(map_x).astype(int), data[y].map(map_y).astype(int)
    y_ = y_ + uniform(-dist, dist, size=len(y_))
    if hue is None:
        x_ = x_ + uniform(-dist, dist, size=len(x_))
    else:
        x_ = x_ + x_noise_hue(data, x, y, hue, dist)
    hue = data[hue] if hue is not None else hue
    sns.scatterplot(x=x_, y=y_, s=1, ax=ax, hue=hue, **kwargs)

def scatterplot_quali(data:pd.DataFrame=None, x=None, y=None, hue=None, ax=None, **kwargs):
    hue_name = None
    if data is None:
        series = {x.name: x, y.name: y}
        if hue is not None:
            series[hue.name] = hue
        hue_name = None if hue is None else hue.name
        data = pd.DataFrame(series)
        return scatterplot_quali_(data, x.name, y.name, hue_name, ax=ax, **kwargs)
    else:
        return scatterplot_quali_(data, x, y, hue, ax=ax, **kwargs)


    
if __name__ == '__main__':
    fig, ax = plt.subplots()
    diamonds = sns.load_dataset("diamonds")
    scatterplot_quali(data=diamonds, x="cut", y="clarity", ax=ax)
    # scatterplot_quali(x=diamonds["cut"], y=diamonds["clarity"], ax=ax)
    plt.show()