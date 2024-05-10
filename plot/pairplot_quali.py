from math import sqrt
import seaborn as sns
from contingencyplot import contingencyplot
from pingouin import chi2_independence
import pandas as pd
from plot import coefplot


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

    grid = sns.PairGrid(data, hue=hue, vars=categorical_vars)
    grid.map_lower(contingencyplot, color=color, s=s, hue=(data[hue] if hue is not None else hue))
    if hue is None:
        grid.map_diag(sns.histplot, color=color)  # Histograms at the diagonal
    else:
        grid.map_diag(sns.histplot, color=color, element="poly")
    grid.map_upper(coefplot, fg_color=color, coef_func=contingence_coefficient)
    return grid


def contingence_coefficient(x: pd.Series, y: pd.Series):
    data = pd.concat([x, y], axis=1)
    # noinspection PyTypeChecker
    _, _, chi2 = chi2_independence(data, x.name, y.name)
    chi2 = chi2.loc[0, 'chi2']
    return sqrt(chi2 / (chi2 + len(data) * len(data.columns)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    diamond = sns.load_dataset("diamonds")
    g = pairplot_quali(diamond, hue='clarity', color=(.5, .5, .5))
    plt.show()
