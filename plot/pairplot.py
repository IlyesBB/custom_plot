from plot import (contingencyplot, coefplot, pearson_coefficient, contingence_coefficient,
                  lower_plot_quanti, lower_plot_quali)
from pingouin import welch_anova
import pandas as pd
import seaborn as sns
import matplotlib as mlp


def pairplot(data: pd.DataFrame, hue: str | None = None, vars: [str] = None,
             color: tuple[float, ...] | str = (.7, .7, 0), s: int = 5, density=False, cmap='Greens', palette='Set1',
             bins: int = 20):
    """
    @param data: Dataframe containing the variables. Only quantitative variables are kept
    @param hue: Categorical variable name to distinguish points with
    @param vars: List of variable names to display
    @param color: Used for markers if hue=None
    @param s: Marker size
    @param density: Whether to display the points or their density
    @param palette: color map for densities
    @param cmap: color map for densities
    @param bins: Bins for bivariate histograms
    """
    if vars is None:
        vars = data.columns
    hue_order = data[hue].cat.categories if hue is not None else None
    grig = sns.PairGrid(data, hue=hue, hue_order=hue_order, vars=vars, diag_sharey=False)
    hue = hue if hue is None else data[hue]
    grig.map_lower(lower_plot, hue=hue, c=color, s=s, density=density, cmap=cmap, palette=palette, bins=bins,
                   stranger=1)
    grig.map_diag(sns.histplot, color=color, hue=hue, hue_order=hue_order, palette=palette, multiple='stack')
    grig.map_upper(upper_plot)
    return grig


def lower_plot(x: pd.Series, y: pd.Series, hue=None, hue_order=None, c: tuple[float, ...] | str = (.7, .7, 0),
               s: int = 5, density=False, cmap='Greens', palette='Set1', bins: int = 20, **kwargs):
    del kwargs['stranger']
    color = c
    if isinstance(x.dtype, pd.CategoricalDtype):
        if isinstance(y.dtype, pd.CategoricalDtype):
            if density:
                return lower_plot_quali(x, y, color=color, density=density, cmap=cmap, hue=hue, hue_order=hue_order, palette=palette, **kwargs)
            else:
                return lower_plot_quali(x, y, color=color, s=s, hue=hue, density=density, palette=palette, hue_order=hue_order, **kwargs)
        if density:
            norm = mlp.colors.LogNorm()
            return sns.histplot(x=x, y=y, cmap=cmap, vmin=None, vmax=None, norm=norm, bins=bins, hue=hue,
                                palette=palette, hue_order=hue_order, **kwargs)
        else:
            return contingencyplot(x=x, y=y, hue=hue, hue_order=hue_order, color=color, s=s, **kwargs)
    elif isinstance(y.dtype, pd.CategoricalDtype):
        if density:
            norm = mlp.colors.LogNorm()
            return sns.histplot(x=x, y=y, cmap=cmap, vmin=None, vmax=None, norm=norm, bins=bins, hue=hue,
                                palette=palette, hue_order=hue_order, **kwargs)
        else:
            return contingencyplot(x=x, y=y, hue=hue, hue_order=hue_order, color=color, s=s, **kwargs)
    else:
        if not density:
            return lower_plot_quanti(x, y, color=color, s=s, density=density, palette=palette, hue=hue, hue_order=hue_order, **kwargs)
        else:
            return lower_plot_quanti(x, y, color=color, density=density, palette=palette, hue=hue, hue_order=hue_order, cmap=cmap, bins=bins,
                              **kwargs)


def upper_plot(x: pd.Series, y: pd.Series, hue=None, **kwargs):
    if isinstance(x.dtype, pd.CategoricalDtype):
        if isinstance(y.dtype, pd.CategoricalDtype):
            coef_func = contingence_coefficient
        else:
            coef_func = eta2_coefficient
    elif isinstance(y.dtype, pd.CategoricalDtype):
        coef_func = eta2_coefficient
    else:
        coef_func = pearson_coefficient
    return coefplot(x, y, hue=hue, coef_func=coef_func, **kwargs)


def eta2_coefficient(x: pd.Series, y: pd.Series):
    if isinstance(x.dtype, pd.CategoricalDtype):
        y, x = x, y
    data = pd.concat([x, y], axis=1)
    # noinspection PyTypeChecker
    res = welch_anova(dv=x.name, between=y.name, data=data)
    return res['np2'].iloc[0]


def map_lower():
    pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    diamond = sns.load_dataset("diamonds")
    print(diamond.head())
    vars = ['clarity', 'color', 'price', 'carat', 'cut']
    pairplot(diamond, vars=None, density=True, color='green', cmap='Greens', palette='Set1', hue='cut')
    plt.show()
