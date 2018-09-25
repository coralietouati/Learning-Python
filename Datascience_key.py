
from datascience import *
from math import *
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# How to use table

values = make_array(1,2,2,3,3,3,4,4,5,8,9,12,5,8,4,9,10,11)
sizes = values**2 -2
ages = np.random.randint(0,60,18) # make a random array
classe = make_array('no','no','yes','yes','yes','no','yes','yes','no','yes','no','no','no','yes','no','yes','yes','no')
values_table = Table().with_columns('Values', values)
values_table = values_table.with_columns('Sizes', sizes)
values_table = values_table.with_columns('Classe', classe)
values_table = values_table.with_columns('Age', ages)
values_table.num_rows

values_table.labels

# select
values_table.show(2)
values_table.take(1) # return a table of the second rowle deuxieme
values_table.take([0,2]) # les deux premiers
example = values_table.drop('Classe').take(10) # type table
row = values_table.drop('Classe').row(10) # type row
values_table.get([1,2])
values_table.get(np.arange(2))
# return array of the column 1
values_table.get('Sizes')
values_table.where('Classe','yes')
values_table.where('Sizes',are.above(3))
values_table.get()
# sort

# How to plot
bins_display = np.arange(0,8,1) # from 0.5 to 10 by step=1
values_table.hist('Values',group='Classe', bins = bins_display, ec='w')
plt.savefig('histo.png')

values_table.scatter('Values','Age',colors= 'Classe')
plt.savefig('scatter.png')

values_table.to
# 3D plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)
ax.scatter(values_table.column('Values'),
           values_table.column('Sizes'),
           values_table.column('Age'),
           c = values_table.column('Classe'),
           cmap = 'viridis');
plt.savefig('3D.png')

#Sample
sample = values_table.sample(10, with_replacement=False)
sample_def = values_table.sample() # creating a new sample, same size but with replacement
sample.num_rows
sample_def.num_rows

# group by
values_table.group('Classe', np.mean)
values_table.group('Classe') # will count number of element in each category
values_table.index_by('Values') # tri syr valye et donne la valeur comme index

#Caculations
values_table.drop('Classe').apply(sum) # sum rows of all columns
values_table.column('Sizes').sum() # sum of columns

np.take(values,np.arange(2,4)) # take the values from index 2 to 4 in a array

values[0]+1
np.val

k=0
words =[]
count = []
words+= 'hjj'
words