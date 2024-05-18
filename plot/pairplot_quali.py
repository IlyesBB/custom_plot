from math import sqrt
import seaborn as sns
from contingencyplot import contingencyplot
from pingouin import chi2_independence
import pandas as pd
from plot import coefplot


def pairplot_quali(data: pd.DataFrame, hue: str = None, color=(.7, .7, 0), s=1, density=False, palette='Set1',
                   cmap='Greens'):
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
    @param density: Whether to display the points or their density
    @param palette: color map for contingency plot and diagonal histograms
    @param cmap: Color map for heatmap
    """
    categorical_vars = data.select_dtypes('category').columns

    grid = sns.PairGrid(data, hue=hue, vars=categorical_vars, palette=palette, diag_sharey=False)
    hue = hue if hue is None else data[hue]
    if density:
        # Plotting heatmaps
        ###################
        grid.map_lower(lower_plot, color=color, density=density, cmap=cmap)
    else:
        # Plotting contingency plots
        #############################
        grid.map_lower(lower_plot, color=color, s=s, density=density)
    grid.map_diag(sns.histplot, color=color, hue=hue, multiple='stack')
    hue = hue if not density else None
    grid.map_upper(coefplot, hue=hue, fg_color=color, coef_func=contingence_coefficient)
    return grid


def lower_plot(x: pd.Series, y: pd.Series, hue: pd.Series = None, density: bool = False, **kwargs):
    """
    Display either a heatmap or a contingency plot, dependent on 'density' kwarg
    Any additional kwarg is passed to heatmap (hue not included) or contingencyplot

    Made to be used in Seaborn PairGrid.map_lower method

    Note: If hue not explicitly in signature, will be called for each value of hue when specified otherwise

    @param x: Abscissa categorical series
    @param y: Ordinate categorical series
    @param hue: Categorical series to color points with
    @param density: Whether to display a heatmap (True) or a contingency plot (False)
    """
    if density:
        # Plotting heatmap
        ###################
        if 'hue_order' in kwargs.keys():
            del kwargs['hue_order']
        del kwargs['palette']
        cross_tab = pd.crosstab(y, x, margins=False)
        # TODO: Make plot start at 0 instead of 0.5
        return sns.heatmap(cross_tab, norm='log', cbar=False, **kwargs)
    else:
        # Plotting contingency plot
        #############################
        return contingencyplot(x=x, y=y, hue=hue, **kwargs)


def contingence_coefficient(x: pd.Series, y: pd.Series) -> float:
    """
    Calculates the contingency coefficient between 2 categorical variables
    @param x: Abscissa categorical series
    @param y: Ordinate categorical series
    @return: Contingency coefficient
    """
    data = pd.concat([x, y], axis=1)
    # noinspection PyTypeChecker
    _, _, chi2 = chi2_independence(data, x.name, y.name)
    chi2 = chi2.loc[0, 'chi2']
    return sqrt(chi2 / (chi2 + len(data) * len(data.columns)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    diamond = sns.load_dataset("diamonds").iloc[:1000]
    g = pairplot_quali(diamond, hue='cut', color='green', palette='Set1', density=False, s=5)
    plt.show()
