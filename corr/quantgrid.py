import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
import seaborn as sns
from scipy.stats import pearsonr


def quantgrid(data, hue=None, vars=None, x_vars=None, y_vars=None, hue_order=None, palette=None, hue_kws=None,
             corner=False, diag_sharey=True, height=2.5, aspect=1, layout_pad=0.5, despine=True, dropna=False,
             r2_cmap='RdBu_r', r2_bg=(0, 0, 0, 1), r2_fg=(0, 0, 0, .1), color=(.7, .7, 0)):
    """
    Displays a correlation grid as a mix of pair plot and heatmap.

    This functions uses Seaborn PairGrid object and uses the same arguments, with a few more.
    Using corrgrid you can display:
    - Scatter plots at the bottom for each pair of variables;
    - Kernel density estimates at the diagonal for each variable;
    - Heatmap at the top, representing the Pearson's coefficient of each pair of variables.

    If hue is initialized, r2 importance is represented with a rectangle sizes (size map) instead of colors (heatmap).

    r2_cmap: Only used when hue is None. Name of color palette to display r2 coefs
    r2_bg: Only used when hue is initialized. Background around the rectangle.
    r2_fg: Only used when hue is initialized. Color of the rectangle.
    color: Color for the diagonal plots (and the lower plots if hue is None)
    """
    g = sns.PairGrid(data, hue=hue, vars=vars, x_vars=x_vars, y_vars=y_vars, hue_order=hue_order, palette=palette,
                     hue_kws=hue_kws, corner=corner, diag_sharey=diag_sharey, height=height, aspect=aspect, 
                     layout_pad=layout_pad, despine=despine, dropna=dropna)
    g.map_lower(sns.scatterplot, color=color)  # Scatter plots at the botton
    g.map_diag(sns.kdeplot, hue=None, color=color)  # Kernel densities at the diagonal
    g.map_upper(corrplot, bg_color=r2_bg, fg_color=r2_fg, cmap=r2_cmap)  # Heatmap or sizemap at the top, depending on hue
    return g

def corrplot(x, y, hue=None, bg_color=(0, 0, 0, 1), fg_color=(0, 0, 0, .1), cmap='RdBu_r', ax=None, **kwargs):
    """
    Displays the Pearson coefficient between x and y as a note in the middle of a rectangle.

    If hue is None, the rectangle color will be linked to R2. Otherwise, Its size will be linked to R2.

    bg_color: Background color around the rectangle. Only used when hue is set i.e the rectangle lengths are proportional to R2.
    fg_color: Color of the rectangle. Only used when hue is set.
    cmap: Used as a color scale to map coefficients to colors. Only used when hue is None.
    """
    # Getting plot infos
    axis = plt.gca() if ax is None else ax
    x_min, x_max, y_min, y_max = plt.axis()
    x_spread, y_spread = (x_max - x_min), (y_max - y_min)

    # Adding the r2 coef as a note
    r2, pval = pearsonr(x,y)
    axis.annotate('%.2f' % r2, xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    axis.axis('off')

    # Setting the rectangle width, height and color
    if hue is None:  # Rectangle color linked to E2
        width, height = x_spread, y_spread
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        fg_color = mpl.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(r2)
    else:  # Rectangle size linked to R2
        width, height = max(abs(r2), .2)*x_spread*.9, max(abs(r2), .2)*y_spread*.9
        bg_rect = Rectangle((x_min, y_min), x_spread, y_spread, fill=False, color=bg_color)
        axis.add_patch(bg_rect)
    # Adding the foreground rectangle
    x_center, y_center = (x_max+x_min)/2, (y_max+y_min)/2
    r2_rect = Rectangle((x_center-width/2, y_center-height/2), width, height, fill=True, color=fg_color)
    axis.add_patch(r2_rect)

if __name__ == '__main__':
    diamonds = sns.load_dataset("diamonds")
    hue='clarity'
    flexible = True
    if not flexible:
        if hue:
            g = quantgrid(diamonds, hue=hue, color=(.5, .5, .5))
        else:
            g = quantgrid(diamonds, color=(.7, .7, 0))
    else:
        g = sns.PairGrid(diamonds, hue=hue, diag_sharey=False)
        g.map_lower(sns.scatterplot, color=(.7, .7, 0), s=.3)
        g.map_diag(sns.kdeplot, color=(.3, .3, .3))
        g.map_upper(corrplot, cmap='Spectral')
    g.fig.set_size_inches(9,7)
    
    plt.show()