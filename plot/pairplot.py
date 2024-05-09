import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd


def pairplot(data: pd.DataFrame, hue: str | None = None, color: tuple[float, ...] = (.7, .7, 0), s: int = 5):
    """
    Similar to pair plot seaborn function, but upper diagonal graphs are made with this module's pearson_plot.

    This plot consists of:
    - Scatter plots at the bottom diagonal;
    - Kernel density estimates at the diagonal;
    - Heatmap or sizemap at the top diagonal, representing the Pearson's coefficients.

    @param data: Dataframe containing the variables. Only quantitative variables are kept
    @param hue: Categorical variable name to distinguish points with
    @param color: Used for markers if hue=None
    @param s: Marker size
    """
    grig = sns.PairGrid(data, hue=hue)
    grig.map_lower(sns.scatterplot, color=color, s=s)
    grig.map_diag(sns.kdeplot, hue=None, color=color)
    grig.map_upper(pearsonplot, hue=hue)
    return grig


def pearsonplot(x: pd.Series, y: pd.Series, hue: str | None = None, bg_color: tuple[float, ...] = (0, 0, 0, 1),
                fg_color: tuple[float, ...] = (0, 0, 0, .1), cmap: str = 'RdBu_r', ax: plt.axis = None):
    """
    Displays the Pearson coefficient between x and y as a label in a rectangle.

    If hue is None, the rectangle color will be linked to R2, with respect to cmap.
    Else, the rectangle size will be linked to R2.

    @param x: Abscissa quantitative variable series
    @param y: Ordinate quantitative variable series
    @param hue: Categorical variable name to distinguish points with
    @param bg_color: Background color around the R2 rectangle. (hue != None)
    @param fg_color: Color of the R2 rectangle. (hue != None)
    @param cmap: Color scale of to map R2 coefficients to rectangle colors. (hue = None)
    @param ax: Matplotlib axis on with add the R2 plot
    """
    ax = plt.gca() if ax is None else ax

    # Adding the r2 coefficient as a label
    #######################################
    r2, p_val = pearsonr(x, y)
    ax.annotate('%.2f' % r2, xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    ax.axis('off')

    # Setting the rectangle around R2 coefficient
    ###############################################
    x_min, x_max, y_min, y_max = plt.axis()
    x_spread, y_spread = (x_max - x_min), (y_max - y_min)
    if hue is None:
        # Rectangle size is maximal, and its color is mapped from R2 with respect to cmap
        width, height = x_spread, y_spread
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        fg_color = mpl.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(r2)
    else:
        # Rectangle size is mapped from R2, it's color is constant and it has a background
        width, height = max(abs(r2), .2) * x_spread * .9, max(abs(r2), .2) * y_spread * .9
        bg_rect = Rectangle((x_min, y_min), x_spread, y_spread, fill=False, color=bg_color)
        ax.add_patch(bg_rect)

    # Adding the foreground rectangle
    #################################
    x_center, y_center = (x_max + x_min) / 2, (y_max + y_min) / 2
    r2_rect = Rectangle((x_center - width / 2, y_center - height / 2), width, height, fill=True, color=fg_color)
    ax.add_patch(r2_rect)


if __name__ == '__main__':
    diamond = sns.load_dataset("diamonds").iloc[:5000]
    g = pairplot(diamond, hue='clarity', color=(.5, .5, .5))
    plt.show()
