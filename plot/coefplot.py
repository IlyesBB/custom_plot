from typing import Callable
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
from matplotlib.patches import Rectangle


def coefplot(x: pd.Series, y: pd.Series, hue: str | None = None, bg_color: tuple[float, ...] = (0, 0, 0, 1),
             fg_color: tuple[float, ...] = (0, 0, 0, .1), cmap: str = 'RdBu_r', ax: plt.axis = None,
             coef_func: Callable[[pd.Series, pd.Series], float] = None, **kwargs):
    """
    Displays an association coefficient between x and y as a label in a rectangle.

    If hue is None, the rectangle color will be linked to the coefficient, with respect to cmap.
    Else, the rectangle size will be linked to the coefficient.

    @param x: Abscissa variable series
    @param y: Ordinate variable series
    @param hue: Categorical variable name to distinguish points with
    @param bg_color: Background color around the coefficient rectangle. (hue != None)
    @param fg_color: Color of the coefficient rectangle. (hue != None)
    @param cmap: Color scale of to map coefficients to rectangle colors. (hue = None)
    @param ax: Matplotlib axis on with add the coefficient plot
    @param coef_func: Function to calculate coefficient
    """
    ax = plt.gca() if ax is None else ax

    # Adding the coefficient as a label
    #######################################
    coef = coef_func(x, y)
    ax.annotate('%.2f' % coef, xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    ax.axis('off')

    # Setting the rectangle around coefficient
    ###############################################
    x_min, x_max, y_min, y_max = plt.axis()
    x_spread, y_spread = (x_max - x_min), (y_max - y_min)
    if hue is None:
        # Rectangle size is maximal, and its color is mapped from coefficient with respect to cmap
        width, height = x_spread, y_spread
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        # noinspection PyTypeChecker
        fg_color = mpl.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(coef)
    else:
        # Rectangle size is mapped from R2, it's color is constant and it has a background
        width, height = max(abs(coef), .2) * x_spread * .9, max(abs(coef), .2) * y_spread * .9
        bg_rect = Rectangle((x_min, y_min), x_spread, y_spread, fill=False, color=bg_color)
        ax.add_patch(bg_rect)

    # Adding the foreground rectangle
    #################################
    x_center, y_center = (x_max + x_min) / 2, (y_max + y_min) / 2
    coef_rect = Rectangle((x_center - width / 2, y_center - height / 2), width, height, fill=True, color=fg_color)
    ax.add_patch(coef_rect)
