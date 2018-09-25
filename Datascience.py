
from datascience import *
from math import *
import numpy as np
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

#plt.style.use('fivethirtyeight')
plt.interactive(True)

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

###NEW TO ADD

x = np.arange(1, 7, 1)
y = make_array(2, 3, 1, 5, 2, 7)
t = Table().with_columns(
        'x', x,
        'y', y
    )
t
t.scatter('x', 'y', s=30, color='red')
plt.show()

t = t.with_column(' x st units', standard_units(x), 'y st units', standard_units(y))

df = pd.DataFrame('xx' [1,2,3],
                'yy': [4,6,7])

df= pd.DataFrame([])
df['xx'] = x
df['yy'] = y

sns.scatterplot(x='xx',y='yy',data=df)
plt.show()

faithful = Table()
valduration = make_array(3.6,1.8,3.332,.283,4.5)
valwait = make_array(79,54,74,62,85)

faithful = faithful.with_columns('duration', valduration, 'wait', valwait)
faithful.scatter('duration', 'wait')
plt.show()

duration_mean = np.mean(faithful.column('duration'))
duration_std = np.std(faithful.column('duration'))
wait_mean = np.mean(faithful.column('wait'))
wait_std = np.std(faithful.column('wait'))

faithful_standard = Table().with_columns(
    "duration (standard units)", (faithful.column('duration')-duration_mean)/duration_std,
    "wait (standard units)",(faithful.column('wait')-wait_mean)/wait_std)
faithful_standard
faithful_standard.hist(overlay=False)
plt.savefig('test2.png')
faithful_standard.column(0)


def simulate_sample_mean(table, label, sample_size, repetitions):
    means = []

    for i in np.arange(repetitions):
        new_sample = table.column(label).sample(sample_size)
        new_sample_mean = np.mean(new_sample)
        means = np.append(means, new_sample_mean)

    sample_means = Table().with_column('Sample Means', means)

simulate_sample_mean(faithful,'wait',4,2)



### ADD 8/9
def correlation(t,x,y):
     x_su = standard_units(t.column(x))
     y_su = standard_units(t.column(y))
     return np.average(x_su*y_su)

def slope(t,x,y):
     r = correlation(t,x,y)
     return r * np.std(t.column(y))/np.std(t.column(y))

def intercept(t,x,y):
     a = slope(t,x,y)
     return np.average(t.column(y)) - a*np.average(t.column(x))

t_slope = slope(table,'Age','weight')
t_inter = intercept(table,'Age','weight')
table.scatter('Age','weight')
plt.show()

def fitted_values(t,x,y):
     a = slope(t,x,y)
     b = intercept(t,x,y)
     return a * t.column(x) + b

val = fitted_values(table, 'Age', 'weight')

table = table.with_column('fittedval', val)

corr = 0.262079
a = corr * 752.475/0.464763
a2 = 1/a
a3 = corr * 0.464763/752.475
b2 = 0.919016 - a3 * 1010.4


# least square mean
def lw_mse(anyslope, anyintercept):
    x = table.column(0)
    y = table.column(1)
    predict = anyslope* x + anyintercept
    return np.mean((y-predict)**2)

####8/15
def plot_data_and_line(dataset, x, y, point_0, point_1):
    """Makes a scatter plot of the dataset, along with a line passing through two points."""
    dataset.scatter(x, y, label="data")
    xs, ys = zip(point_0, point_1)
    plots.plot(xs, ys, label="regression line")
    plots.legend(bbox_to_anchor=(1.5,.8))

plot_data_and_line(faithful_standard,
                   "duration (standard units)",
                   "wait (standard units)",
                   [-2, -2*r],
                   [2, 2*r])

# Tracer une line definit par deux points
plt.plot([3, 3], [40, 100])
plt.show()

table.where(table.column(1),are.below(5))

np.random.normal(0,1,5)

sizes = make_array(12, 17, 6, 9, 7)
np.mean(sizes)
np.sort(sizes)
percentile(10, sizes)
countries = Table().with_column('size', sizes)
countries.hist(bins = np.arange(0,20,1))
plt.savefig('countries')
len(sizes)*0.7



def errors(t, slope, intercept):
    x = t.column(0)
    y = t.column(1)
    y_p = slope * x + intercept
    return y - y_p

def lsr(slope, intercept):
    x = t.column(0)
    y = t.column(1)
    y_p = slope * x + intercept
    return np.mean((y - y_p)**2)

def fit_line(tbl):
    # Your code may need more than 1 line below here.
    # Rather than using the regression line formulas, try
    # calling minimize on the mean squared error.
    t=tbl
    x = t.column(0)
    y = t.column(1)
    def mse(slope, intercept):
        y_p = slope * x + intercept
        return np.mean((y - y_p) ** 2)

    fit = minimize(mse)
    slope = fit[0]
    intercept = fit[1]
    return make_array(slope, intercept)


fit_line(faithful)

# Here is an example call to your function.  To test your function,
# figure out the right slope and intercept by hand.
example_table = Table().with_columns(
    "Speed (parsecs/year)", make_array(0, 1),
    "Distance (million parsecs)", make_array(1, 3))
fit_line(example_table)

minimize(errors)

# doctor

ckd = Table.read_table('https://www.inferentialthinking.com/notebooks/ckd.csv')

## K nearest neighbor
def distance(pt1, pt2):
    """Return the distance between two points (represented as arrays)"""
    return np.sqrt(sum((pt1 - pt2) ** 2))

def row_distance(row1, row2):
    """Return the distance between two numerical rows of a table"""
    return distance(np.array(row1), np.array(row2))


def distances(training, example):
    """Compute a table with the training set and distances to the example for each row in the training set."""
    dists = []
    attributes = training.drop('Classe')
    for row in attributes.rows:
        dist = row_distance(row, example)
        dists.append(dist)
    return training.with_column('Distance', dists)

def closest(training, example, k):
    """Return a table of the k closest neighbors to example"""
    return distances(training,example).sort('Distance').take(np.arange(k))


def majority_class(neighbors):
    """Return the class that's most common among all these neighbors."""
    #option 1 (with the argmax equivalent to idxmax)s
    #index = nn.group('Classe').column('count').argmax()
    #nn.group('Classe').column('Classe').item(index)
    return neighbors.group('Classe').sort('count', descending=True).column('Classe').item(0)

def classify(training, example, k):
    "Return the majority class among the k nearest neighbors."
    nearest_neighbors = closest(training, example, k)
    return majority_class(nearest_neighbors)

def evaluate_accuracy(training, test, k):
    test_attributes = test.drop('Classe')
    num_correct = 0
    for i in np.arange(test.num_rows):
        # Run the classifier on the ith patient in the test set
        test_patient = test_attributes.row(i)
        c = classify(training, test_patient, k)
        # Was the classifier's prediction correct?
        if c == test.column('Classe').item(i):
            num_correct = num_correct + 1
    return num_correct / test.num_rows


example = values_table.drop('Classe').row(10)
training = values_table.exclude(12)
nn = closest(training,example,3)
majority_class(nn)

# evaluation

# Suffle the training
nb_rows=int(values_table.num_rows)
shuffled = values_table.sample(with_replacement=False)
training_set = shuffled.take(np.arange(0,int(nb_rows/2)))
test_set = shuffled.take(np.arange(int(nb_rows/2),nb_rows))

