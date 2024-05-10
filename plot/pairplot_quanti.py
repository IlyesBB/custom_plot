import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd
from plot import coefplot

def pairplot_quanti(data: pd.DataFrame, hue: str | None = None, color: tuple[float, ...] = (.7, .7, 0), s: int = 5):
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
    if hue is None:
        grig.map_diag(sns.histplot, hue=None, color=color)
    else:
        grig.map_diag(sns.kdeplot, fill=True)
    grig.map_upper(coefplot, hue=hue, coef_func=pearson_coefficient)
    return grig


def pearson_coefficient(x: pd.Series, y: pd.Series):
    r2, p_val = pearsonr(x, y)
    return r2


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    diamond = sns.load_dataset("diamonds").iloc[:1000]
    g = pairplot_quanti(diamond, hue='clarity', color=(.5, .5, .5))
    plt.show()
