import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
from matplotlib import cm, colors, rc
import matplotlib as mpl


def bezier_curve(x: pd.Series, y: pd.Series, c: np.ndarray):
    control_x = np.linspace(x.iloc[0], x.iloc[-1], len(y) * 3 - 2, endpoint=True)
    control_y = np.repeat(y, 3)[1:-1]
    control_pts = list(zip(list(control_x), control_y))
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(control_pts) - 1)
    path = Path(control_pts, codes)
    return patches.PathPatch(path, facecolor='none', lw=1, edgecolor=c)


def axis_for_each_col(data: pd.DataFrame, ax: plt.Axes) -> [plt.Axes]:
    col_min, col_max = data.min(axis=0), data.max(axis=0)
    col_spread = col_max - col_min

    axes = [ax] + [ax.twinx() for _ in range(len(data.columns) - 1)]
    for i, axis in enumerate(axes):
        axis.set_ylim(col_min.iloc[i] - 0.05 * col_spread.iloc[i], col_max.iloc[i] + 0.05 * col_spread.iloc[i])
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        if axis != ax:
            axis.spines['left'].set_visible(False)
            axis.yaxis.set_ticks_position('right')
            axis.spines["right"].set_position(("axes", i / (len(data.columns) - 1)))

    ax.set_xlim(0, len(data.columns) - 1)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns)
    ax.spines['right'].set_visible(False)
    ax.xaxis.tick_top()
    return axes


def parallelplot(data: pd.DataFrame, hue: str, cmap: str = None, ax: plt.Axes = None, bezier=True):
    """
    Creates a parallel coordinates plot.

    Field values are scaled to first field, and plotted in first axis.

    @param data: A list containing one array per parallel axis, each containing N data points.
    @param hue: The field name to distinguish rows with. Can be a categorical or a numeric field.
    @param cmap: Name of the color map
    @param ax: Axis on which draw the plot
    @param bezier: Whether draw Bezier cubic curves or straight lines
    """
    # Selecting quantitative variables
    ###################################
    hue = data[hue]
    data = data[data.columns.difference(pd.Index([hue.name]))].copy()  # Removing hue
    data = data.select_dtypes('number').astype(float)

    # Initializing axes and color mapper
    #####################################
    ax = ax if ax is not None else plt.subplots()[1]
    axes = axis_for_each_col(data, ax)
    if cmap is None:
        cmap = 'Set1' if isinstance(hue.dtype, pd.CategoricalDtype) else 'Blues'
    hue_min, hue_max = (1, hue.nunique()) if isinstance(hue.dtype, pd.CategoricalDtype) else (hue.min(), hue.max())
    norm = colors.Normalize(vmin=hue_min, vmax=hue_max, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap])

    # Transform data to fit first axis
    ###################################
    col_min, col_max = data.min(axis=0), data.max(axis=0)
    col_spread = col_max - col_min
    data.iloc[:, 1:] = (data.iloc[:, 1:] - col_min.iloc[1:]) / col_spread.iloc[1:]
    data.iloc[:, 1:] = col_min.iloc[0] + data.iloc[:, 1:] * col_spread.iloc[0]

    # Drawing lines with respect to first axis
    ###########################################
    for ind_row in range(len(data)):
        hue_val = hue.cat.codes.iloc[ind_row] if isinstance(hue.dtype, pd.CategoricalDtype) else hue.iloc[ind_row]
        x, y, color = pd.Series(range(len(data.columns))), data.iloc[ind_row, :], mapper.to_rgba(hue_val)
        ax.plot(x, y, c=color) if not bezier else ax.add_patch(bezier_curve(x, y, c=color))

    # Adding legend or color scale
    ################################
    if isinstance(hue.dtype, pd.CategoricalDtype):
        plot_hue = pd.DataFrame(zip(ax.get_children(), hue)).drop_duplicates(subset=[1])
        ax.legend(plot_hue[0], plot_hue[1], loc='lower center', bbox_to_anchor=(0.5, -0.18),
                  ncol=len(plot_hue), fancybox=True, shadow=True)
    else:
        plt.colorbar(mapper, ax=ax, location='right', label=hue.name)


if __name__ == '__main__':
    import seaborn as sns

    mpg = sns.load_dataset('mpg')
    plt.title('Cars caracteristics over time')

    parallelplot(mpg, hue='model_year', bezier=True)

    plt.title('Cars caracteristics over time')

    plt.show()
