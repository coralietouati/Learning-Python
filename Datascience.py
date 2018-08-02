
from datascience import *
from math import *
import numpy as np
from scipy import stats

#%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#plt.interactive(True)

values = make_array(1,2,2,3,3,3,4,4,5)
sum(values) / len(values)

values_table = Table().with_columns('Values', values)
bins_display = np.arange(0,8,1) # from 0.5 to 10 by step=1
values_table.hist(bins = bins_display, ec='w')
plt.show()

percentile(50, values) # gives the median
values.mean()

# Standard deviation
values = make_array(2,3,3,9,10)
table = Table().with_columns('Age', values)
table
deviations = values- np.average(values)
table = table.with_columns('Deviation', deviations)
table = table.with_columns('squared Deviation', deviations**2)
# Variance
variance = table.column('squared Deviation').mean()
sd = variance ** 0.5
np.std(values)

# chebechov
weightvalues = make_array(1,3,1.2,3.6,4.5)
table = table.with_columns('weight',weightvalues)

plt.rcParams.update({'font.size': 10})
table.hist(overlay=False, ec = 'w')
plt.show()

for k in table.labels:
    val = table.column(k)
    average = np.mean(val)
    sd = np.std(val)
    print('\n' + str(k))
    for z in np.arange(2,6): # same as range(2,6)
        chosen = table.where(k, are.between(average - z*sd, average + z*sd))
        proportion = (chosen.num_rows / table.num_rows)*100
        print('Average +/-', z, 'SDs:', proportion)


#### Bell curve
def standard_units(x):
    return (x - np.mean(x))/np.std(x)

standard_units(weightvalues)
x=np.linspace(-5,5)
plt.plot(x, np.exp(x))
plt.show()

exp(0)