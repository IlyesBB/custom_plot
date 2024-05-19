import pandas as pd
import numpy as np
from numpy.random import uniform
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union


def x_noise_hue(x: pd.Series, y: pd.Series, hue: pd.Series, square_len: float) -> pd.Series:
    """
    Return the noise to add to variable x, once mapped to integers

    @param x: Abscissa categorical variable series
    @param y: Ordinate categorical variable series
    @param hue: Categorical variable to distinguish points with
    @param square_len: Length of the square to plot the intersection of 2 values
    """
    # Creating min and max each x, y, and hue value
    ################################################
    if hue.name not in [x.name, y.name]:
        data = pd.concat([x, y, hue], axis=1)
    else:
        data = pd.concat([x, y], axis=1)
    ptable = pd.pivot_table(data=data.reset_index(), values='index', index=(x.name, y.name), columns=hue.name,
                            aggfunc='count', observed=False)
    row_sums = ptable.sum(axis=1)
    for col in ptable.columns:
        ptable[col] = square_len * ptable[col] / row_sums
    ptable_max = ptable.cumsum(axis=1) - square_len / 2
    ptable_min = ptable_max.shift(periods=1, axis=1).fillna(-square_len / 2)
    # From table to series (adding column as index)
    ptable_max, ptable_min = ptable_max.stack(), ptable_min.stack()
    ptable_max.name, ptable_min.name = 'max', 'min'

    # Adding min/max and noisy x coordinate
    #########################################
    data = pd.merge(data, ptable_min, left_on=(x.name, y.name, hue.name), right_index=True)
    data = pd.merge(data, ptable_max, left_on=(x.name, y.name, hue.name), right_index=True)
    data['diff'] = data['max'] - data['min']
    data['rand'] = np.random.uniform(0, 1, size=len(data))
    return data['rand'] * data['diff'] + data['min']


# noinspection PyUnboundLocalVariable
def contingencyplot(x: pd.Series, y: pd.Series, hue: pd.Series = None, ax: plt.axis = None, square_len=0.5, **kwargs):
    """
    Scatter plot between 2 categorical variables

    Made to be used with PairGrid, as, on its own, it's not much different than a contingency table

    @param x: Abscissa categorical variable series
    @param y: Ordinate categorical variable series
    @param hue: Categorical variable name to distinguish points with
    @param ax: Matplotlib axis to witch add plot
    @param square_len: Length of the square to plot the intersection of 2 values
    """
    # Mapping categorical variables to integers
    ###########################################
    # distinct_x, distinct_y: Distinct values for x and y
    change = False
    if x.dtype.name != 'category':
        change = True
        x, y = y, x

    distinct_x: pd.Index = x.cat.categories
    x_to_int = {val: i for (i, val) in enumerate(distinct_x)}
    x_real = x.map(x_to_int).astype(int)

    # Adding noise to have a "crowd feeling"
    ##########################################
    if hue is None:
        x_real = x_real + uniform(-square_len / 2, square_len / 2, size=len(x_real))
    else:
        x_real = x_real + x_noise_hue(x, y, hue, square_len / 2)
    ax = plt.gca() if ax is None else ax
    if y.dtype.name == 'category':
        distinct_y: pd.Index = y.cat.categories
        y_to_int = {val: i for (i, val) in enumerate(distinct_y)}
        y_real = y.map(y_to_int).astype(int)
        y_real = y_real + uniform(-square_len / 2, square_len / 2, size=len(y_real))
    else:
        y_real = y
    if change:
        # y is numerical
        distinct_y = distinct_x
        y_to_int = x_to_int
        x, y = y, x
        x_real, y_real = y_real, x_real
    if x.dtype.name == 'category':
        ax.set_xticks(range(len(distinct_x)))
        ax.set_xticklabels(sorted([val for val in distinct_x], key=lambda label: x_to_int[label]))
    if y.dtype.name == 'category':
        ax.set_yticks(range(len(distinct_y)))
        ax.set_yticklabels(sorted([val for val in distinct_y], key=lambda label: y_to_int[label]))
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    sns.scatterplot(x=x_real, y=y_real, ax=ax, hue=hue, **kwargs)


if __name__ == '__main__':
    fig, ax_ = plt.subplots()
    diamonds = sns.load_dataset("diamonds").iloc[:10000]
    contingencyplot(diamonds["cut"], diamonds["clarity"], hue=diamonds['color'], ax=ax_, s=8)
    plt.show()
