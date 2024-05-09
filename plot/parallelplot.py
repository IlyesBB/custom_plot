from scipy.stats import rankdata
import seaborn as sns


def parallelplot(data, hue, **kwargs):
    data_percentile = data.copy()
    for col in data_percentile.columns():
        data_percentile[col] = rankdata(data_percentile[col], "average") / len(data_percentile)
    sns.relplot(data=data_percentile, hue=hue, kind='line', **kwargs)
    