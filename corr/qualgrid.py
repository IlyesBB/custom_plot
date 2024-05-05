import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
import seaborn as sns
from scatterplot_quali import scatterplot_quali
from pingouin import chi2_independence
import pandas as pd


def qualgrid(data, hue=None, vars=None, x_vars=None, y_vars=None, hue_order=None, palette=None, hue_kws=None,
             corner=False, diag_sharey=True, height=2.5, aspect=1, layout_pad=0.5, despine=True, dropna=False,
             chi2_cmap='RdBu_r', chi2_bg=(0, 0, 0, 1), chi2_fg=(0, 0, 0, .1), color=(.7, .7, 0)):
    """
    Displays a contingency grid as a mix of scatter plot and heatmap.

    This functions uses Seaborn PairGrid object and uses the same arguments, with a few more.
    Using corrgrid you can display:
    - Scatter plots at the bottom for each pair of variables;
    - Histograms at the diagonal for each variable;
    - Heatmap at the top, representing the p-valueof a chi2 test of independance for each pair of variables

    If hue is initialized, the p-value importance is represented with a rectangle sizes (size map) instead of colors (heatmap).

    chi2_cmap: Only used when hue is None. Name of color palette to display r2 coefs
    chi2_bg: Only used when hue is initialized. Background around the rectangle.
    chi2_fg: Only used when hue is initialized. Color of the rectangle.
    color: Color for the diagonal plots (and the lower plots if hue is None)
    """
    if vars is None:
        vars = data.select_dtypes('category').columns
    g = sns.PairGrid(data, hue=hue, vars=vars, x_vars=x_vars, y_vars=y_vars, hue_order=hue_order, palette=palette,
                     hue_kws=hue_kws, corner=corner, diag_sharey=diag_sharey, height=height, aspect=aspect, 
                     layout_pad=layout_pad, despine=despine, dropna=dropna)
    g.map_lower(scatterplot_quali, color=color)  # Scatter plots at the botton
    if hue is None:
        g.map_diag(sns.histplot, color=color)  # Histograms at the diagonal
    else:
        g.map_diag(sns.histplot, color=color, multiple='fill')
    g.map_upper(chi2plot, bg_color=chi2_bg, fg_color=chi2_fg, cmap=chi2_cmap, color=color)  # Heatmap or sizemap at the top, depending on hue
    return g


def chi2plot(x, y, hue=None, bg_color=(0, 0, 0, 1), fg_color=(0, 0, 0, .1), cmap='RdBu_r', ax=None, **kwargs):
    """
    Displays the p-value of a chi2 test of independance between x and y as a note in the middle of a rectangle.

    If hue is None, the rectangle color will be linked to the p-value. Otherwise, Its size will be linked to the p-value.

    bg_color: Background color around the rectangle. Only used when hue is set i.e the rectangle lengths are proportional to the p-value.
    fg_color: Color of the rectangle. Only used when hue is set.
    cmap: Used as a color scale to map coefficients to colors. Only used when hue is None.
    """
    # Getting plot infos
    axis = plt.gca() if ax is None else ax
    x_min, x_max, y_min, y_max = plt.axis()
    x_spread, y_spread = (x_max - x_min), (y_max - y_min)

    # Adding the r2 coef as a note
    data = pd.DataFrame({x.name: x, y.name:y})
    _, _, chi2 = chi2_independence(data, x.name, y.name)
    chi2 = chi2.loc[0, 'pval']
    axis.annotate('%.3f' % chi2, xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    axis.axis('off')

    # Setting the rectangle width, height and color
    if hue is None:  # Rectangle color linked to E2
        width, height = x_spread, y_spread
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        fg_color = mpl.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(chi2)
    else:  # Rectangle size linked to R2
        width, height = max(abs(chi2), .2)*x_spread*.9, max(abs(chi2), .2)*y_spread*.9
        bg_rect = Rectangle((x_min, y_min), x_spread, y_spread, fill=False, color=bg_color)
        axis.add_patch(bg_rect)
    # Adding the foreground rectangle
    x_center, y_center = (x_max+x_min)/2, (y_max+y_min)/2
    r2_rect = Rectangle((x_center-width/2, y_center-height/2), width, height, fill=True, color=fg_color)
    axis.add_patch(r2_rect)


if __name__ == '__main__':
    diamond = sns.load_dataset("diamonds")
    diamond = diamond.select_dtypes('category')
    hue = True
    flexible = False
    if not flexible:
        if hue:
            g = qualgrid(diamond, hue='cut', color=(.5, .5, .5))
        else:
            g = qualgrid(diamond, color=(.7, .7, 0))
    else:
        g = sns.PairGrid(diamond, hue='cut')
        g.map_lower(sns.scatterplot, color=(.7, .7, 0))
        g.map_diag(sns.kdeplot, color=(.3, .3, .3), hue=None)
        g.map_upper(scatterplot_quali, cmap='Spectral')
    g.fig.set_size_inches(9,7)
    
    plt.show()