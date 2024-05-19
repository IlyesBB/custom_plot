import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd
from multiplot import coefplot
import matplotlib as mlp
import matplotlib.pyplot as plt


def pairplot_quanti(data: pd.DataFrame, hue: str | None = None, color: tuple[float, ...] | str = (.7, .7, 0),
                    s: int = 5, density=False, palette='Set1', cmap='Greens', bins: int = 20):
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
    @param density: Whether to display the points or their density
    @param palette: color map for densities
    @param cmap: Color map for bivariate histograms (2dbins)
    @param bins: Bins for bivariate histograms (2dbins)
    """
    grid = sns.PairGrid(data, hue=hue, diag_sharey=False)
    hue = hue if hue is None else data[hue]
    if not density:
        # Scatter plot
        ###############
        grid.map_lower(lower_plot, color=color, s=s, density=density, palette=palette, hue=hue)
    else:
        # Histogram bivariate (2dbins)
        #############################
        grid.map_lower(lower_plot, color=color, density=density, palette=palette, hue=hue, cmap=cmap, bins=bins)
    grid.map_diag(sns.histplot, hue=hue, palette=palette, color=color, multiple='stack')
    grid.map_upper(coefplot, hue=hue, coef_func=pearson_coefficient)
    return grid


def lower_plot(x: pd.Series, y: pd.Series, hue: pd.Series = None, **kwargs):
    density = kwargs.pop('density')
    if density:
        norm = mlp.colors.LogNorm()
        return sns.histplot(x=x, y=y, vmin=None, vmax=None, norm=norm, hue=hue, **kwargs)
    else:
        return sns.scatterplot(x=x, y=y, hue=hue, **kwargs)


def pearson_coefficient(x: pd.Series, y: pd.Series):
    r2, p_val = pearsonr(x, y)
    return r2


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    diamond = sns.load_dataset("diamonds").iloc[:1000]
    # hue = 'clarity'
    g = pairplot_quanti(diamond, hue=None, color='green', palette='Set1', cmap='Greens', density=False, bins=20)
    plt.show()
