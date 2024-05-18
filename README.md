# Purpose
The aim of this repo is to propose an extension of Seaborn pairplot function, by:
- Allowing it to process categorical data
- Mix it with heatmap for upper diagonal

# Examples
## Scatter plots
```Python
from plot import pairplot
import matplotlib.pyplot as plt
import seaborn as sns

diamond = sns.load_dataset("diamonds").sample(200)
vars = ['clarity', 'color', 'price', 'carat', 'cut']
pairplot(diamond, vars=vars, density=False, color='green', cmap='Greens')

plt.show()
```
![alt text](https://github.com/IlyesBB/custom_plot/blob/master/screenshots/density_false.png?raw=true)


