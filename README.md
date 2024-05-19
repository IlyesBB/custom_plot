# Purpose
The aim of this repo is to propose visualization for multidimentional plots.

It extends the Seaborn pairplot function by:
- Allowing it to process categorical data
- Mix it with heatmap for upper diagonal

Also has a parallel coordinates plot function.

# Examples

## Parallel coordinates plot
```Python
from plot import parallelplot
import matplotlib.pyplot as plt
import seaborn as sns

mpg = sns.load_dataset('mpg')
plt.title('Cars caracteristics over time')

parallelplot(mpg, hue='model_year', bezier=True)
plt.title('Cars caracteristics over time')

plt.show()
```
![alt text](https://github.com/IlyesBB/custom_plot/blob/master/screenshots/parallelplot.png?raw=true)

## Scatter plots
```Python
from plot import pairplot
import matplotlib.pyplot as plt
import seaborn as sns

diamond = sns.load_dataset("diamonds").sample(200)
vars = ['clarity', 'color', 'price', 'carat', 'cut']
pairplot(diamond, vars=vars, density=False, color='green', cmap='Greens')
plt.suptitle('Diamond pairplot')

plt.show()
```
![alt text](https://github.com/IlyesBB/custom_plot/blob/master/screenshots/density_false.png?raw=true)

## Histograms
```Python
from plot import pairplot
import matplotlib.pyplot as plt
import seaborn as sns

diamond = sns.load_dataset("diamonds")
vars = ['clarity', 'color', 'price', 'carat', 'cut']
pairplot(diamond, vars=vars, density=True, color='green', cmap='Greens')
plt.suptitle('Diamond pairplot')

plt.show()
```
![alt text](https://github.com/IlyesBB/custom_plot/blob/master/screenshots/density_true.png?raw=true)

## Histograms with hue
```Python
from plot import pairplot
import matplotlib.pyplot as plt
import seaborn as sns

diamond = sns.load_dataset("diamonds")
vars = ['clarity', 'color', 'price', 'carat', 'cut']
pairplot(diamond, vars=vars, density=True, palette='Set1', hue='cut', color='gray')
plt.suptitle('Diamond pairplot')

plt.show()
```
![alt text](https://github.com/IlyesBB/custom_plot/blob/master/screenshots/density_true_hue.png?raw=true)


# Association coefficients
## Numeric - Numeric
Using Pearson's correlation coefficient

## Categorical - Categorical
Using Pearson's contingency coefficient.

## Categorical - Numeric
One-way ANOVA eta-quared.
