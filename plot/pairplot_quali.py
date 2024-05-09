import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
import seaborn as sns
from contingencyplot import contingencyplot
from pingouin import chi2_independence
import pandas as pd


def pairplot_quali(data: pd.DataFrame, hue: str = None, color=(.7, .7, 0), s=1):
    """
    Similar to pair plot seaborn function, but for categorical variables.

    Scatter plots are .

    This plot consists of:
    - Scatter plots - made with scatterplot_quali - at the bottom diagonal;
    - Kernel density estimates at the diagonal;
    - Heatmap or sizemap at the top diagonal, representing the Pearson's coefficients.

    @param data: Dataframe containing the variables. Only quantitative variables are kept
    @param hue: Categorical variable name to distinguish points with
    @param color: Used for markers if hue=None
    @param s: Marker size
    """
    categorical_vars = data.select_dtypes('category').columns
    categorical_vars = categorical_vars[categorical_vars != hue]
    hue = data[hue] if hue is not None else hue
    grid = sns.PairGrid(data, hue=hue.name, vars=categorical_vars)
    grid.map_lower(contingencyplot, color=color, s=s, hue=hue)  # Scatter plots at the bottom
    if hue is None:
        grid.map_diag(sns.histplot, color=color)  # Histograms at the diagonal
    else:
        grid.map_diag(sns.histplot, color=color, multiple='fill')
    grid.map_upper(chi2plot, fg_color=color)
    return grid


def chi2plot(x: pd.Series, y: pd.Series, hue: pd.Series = None, bg_color: tuple[float, ...] = (0, 0, 0, 1),
             fg_color: tuple[float, ...] = (0, 0, 0, .1), cmap: str = 'RdBu_r', ax: plt.axis = None, **kwargs):
    """
    Displays the p-value of a chi2 test of independence between x and y as a label in a rectangle.

    If hue is None, the rectangle color will be linked to the p-value, with respect to cmap.
    Else, the rectangle size will be linked to the p-value.

    @param x: Abscissa categorical variable series
    @param y: Ordinate categorical variable series
    @param hue: Categorical variable name to distinguish points with
    @param bg_color: Background color around the p-value rectangle. (hue != None)
    @param fg_color: Color of the p-value rectangle. (hue != None)
    @param cmap: Color scale of to map p-value coefficients to rectangle colors. (hue = None)
    @param ax: Matplotlib axis on with add the R2 plot
    """
    axis = plt.gca() if ax is None else ax

    # Adding the p-value as a label
    #######################################
    data = pd.DataFrame({x.name: x, y.name: y})
    _, _, chi2 = chi2_independence(data, x.name, y.name)
    # noinspection SpellCheckingInspection
    chi2 = chi2.loc[0, 'pval']
    axis.annotate('%.3e' % chi2, xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    axis.axis('off')

    # Setting the rectangle around p-value
    ###############################################
    x_min, x_max, y_min, y_max = plt.axis()
    x_spread, y_spread = (x_max - x_min), (y_max - y_min)
    if hue is None:
        # Rectangle size is maximal, and its color is mapped from p-value with respect to cmap
        width, height = x_spread, y_spread
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        fg_color = mpl.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(chi2)
    else:
        # Rectangle size is mapped from p-value, it's color is constant and it has a background
        width, height = max(abs(chi2), .2) * x_spread * .9, max(abs(chi2), .2) * y_spread * .9
        bg_rect = Rectangle((x_min, y_min), x_spread, y_spread, fill=False, color=bg_color)
        axis.add_patch(bg_rect)

    # Adding the foreground rectangle
    #################################
    x_center, y_center = (x_max + x_min) / 2, (y_max + y_min) / 2
    r2_rect = Rectangle((x_center - width / 2, y_center - height / 2), width, height, fill=True, color=fg_color)
    axis.add_patch(r2_rect)


if __name__ == '__main__':
    diamond = sns.load_dataset("diamonds")
    g = pairplot_quali(diamond, hue='clarity', color=(.5, .5, .5))
    plt.show()
