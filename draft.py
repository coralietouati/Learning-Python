import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from Fonctions import *

d = pd.read_csv('C:/Users\ctouati\PycharmProjects\EquinoxStorage-\SPWloads.csv')
d.Date = pd.to_datetime(d.Date)
d.drop(['AddressName','Date_otherformat'], axis=1, inplace=True)
# check if it is week of weekend
d['day'] = pd.DatetimeIndex(d['Date']).dayofweek
d['weekend']= np.where(d.day>=5,'Weekend','Week')
# check the number of data point over the years
count = d.groupby(['Year','Month']).count().load
# look at one year only
#d = d[(d.Date >= "2017-07-01 00:00:00") & (d.Date < "2018-07-01 00:00:00")] # yes or no???


# Look at zip codes, CA 900-961
zips = d.Zip.unique()
zip_notCA =[]
print('Number of zips ' + str(len(zips)))
for zipcode in zips:
    print(zipcode)
    if zipcode>90000 and zipcode<96100:
        print('California')
    else:
        zip_notCA.append(zipcode)
if len(zip_notCA)== 0:
    print('All customer are in CA')



# CLEAN DATA

# Y-a-t-il une valeur nulle ? Si oui combien?
d.load.isnull().any()
d.load.isnull().sum()
d.dropna(subset=['load','ID'], inplace=True) # delete rows with empty load

# Delete high loads --> assumed to be a meter erreur
daily_max = d.groupby(['Hour', 'ID']).max().reset_index()
sns.lineplot(x='Hour', y='load', hue='ID', data=daily_max)
print(str(d[d.load>30].shape[0]) + ' rows were deleted as the value was higher than 30kW')
d = d[d.load<30]
daily_max = d.groupby(['Hour', 'ID']).max().reset_index()
sns.lineplot(x='Hour', y='load', hue='ID', data=daily_max, legend=False)

# OUTLINERS

# keep sites with at least 4380 data points
print(len(d.ID.unique()))
ID_count = d.groupby(['ID']).count().reset_index().sort_values(by=['load'], ascending=False)
print('there are '+ str(ID_count[ID_count.load < 4380].shape[0])+  ' sites that have less than six months of data. Those have been deleted')
ID_count = ID_count[ID_count.load >= 4380]
ID_keep = list(ID_count.ID)
d = d[d.ID.isin(ID_keep)]
print(len(d.ID.unique()))
daily = d.groupby(['Hour', 'ID']).mean().reset_index()
sns.lineplot(x='Hour', y='load', hue='ID', data=daily, legend=False)

# keep those that have at least one value for each hour over the year
print(len(d.ID.unique()))
daily = d.groupby(['Hour', 'ID']).mean().reset_index()
daily_count = daily.groupby(['ID']).count().reset_index()
daily_count = daily_count[daily_count.load==24]
ID_keep = daily_count.ID.unique()
d = d[d.ID.isin(ID_keep)]
print(len(d.ID.unique()))
daily = d.groupby(['Hour', 'ID']).mean().reset_index()
sns.lineplot(x='Hour', y='load', hue='ID', data=daily, legend=False)



# Normalize load
max = d[['ID','load']].groupby(['ID']).max().reset_index()
min = d[['ID','load']].groupby(['ID']).min().reset_index()
d['max'] = 0
d['min'] = 0

for id in d.ID.unique():
    d.loc[d.ID == id, 'max'] = max[max.ID == id].load.item()
    d.loc[d.ID == id, 'min'] = min[min.ID == id].load.item()

d['loadnorm'] = (d.load - d['min']) / (d['max'] - d['min'])

daily = d.groupby(['Hour', 'ID']).mean().reset_index()
sns.lineplot(x='Hour', y='loadnorm', hue='ID', data=daily, legend=False)



#Export in excel
d.to_csv('C:/Users\ctouati\PycharmProjects\EquinoxStorage-\SPWloads_cleaned_only1year.csv')

# REOPEN
d = pd.read_csv('C:/Users\ctouati\PycharmProjects\EquinoxStorage-\SPWloads_cleaned_only1year.csv')
d.Date = pd.to_datetime(d.Date)



# MEAN FOR ALL CUSTOMER

#Look at yearly consumption
consumption = d.groupby(['ID']).mean().reset_index()
consumption.hist('load')
consumption['consumption_kWh'] = consumption.load*365*24
consumption.hist('consumption_kWh')

ID_tokeep= consumption[consumption.load>7000/365/24].ID.unique()
print('ID that have a consumption lower than 7000kWh are being deleted. Total number of ID: ', str(len(consumption[consumption.load<7000/365/24].ID.unique())) )
d = d[d.ID.isin(ID_tokeep)]
consumption = d.groupby(['ID']).mean().reset_index()
consumption.hist('load')
consumption['consumption_kWh'] = consumption.load*365*24
consumption.hist('consumption_kWh')
print('There is ' + str(len(d.ID.unique())) + ' left in the database')



# Global
daily = d.groupby(['ID','Hour']).mean().reset_index()
sns.lineplot(x='Hour', y='load', hue='ID', data=daily, estimator=None, legend=False)
sns.lineplot(x='Hour', y='load', data=d, legend=False)
daily.hist('load', by='Hour')
sns.lineplot(x='Hour', y='loadnorm', hue='ID', data=daily, legend=False)

sns.lineplot(x='Hour', y='loadnorm', hue='ID', data=daily, estimator=None, legend=False)

# get the season cluster!
seasons = cluster(d,'loadnorm','Month',3,'season')


# See the difference, first insight, between week/weekend
week = d.groupby(['ID','weekend','Hour','season']).mean().reset_index()
plt.figure(1)
sns.lineplot(x='Hour', y='loadnorm', hue='weekend', data=week[week.season==1], palette=['grey','mediumblue']).set_title('Winter')
plt.figure(2)
sns.lineplot(x='Hour', y='loadnorm', hue='weekend', data=week[week.season==2], palette=['grey','mediumblue']).set_title('Summer')
plt.figure(3)
sns.lineplot(x='Hour', y='loadnorm', hue='weekend', data=week[week.season==3], palette=['grey','mediumblue']).set_title('Mid')
plt.figure(4)
sns.lineplot(x='Hour', y='loadnorm', hue='weekend', data=week, palette=['grey','mediumblue']).set_title('Year')



# cluster for customer for each season --> get the shape of the profile
customers_winter_norm = cluster(d[d.season==1],'loadnorm','ID',3,'customer_winter_norm')
customers_summer_norm = cluster(d[d.season==2],'loadnorm','ID',4,'customer_summer_norm')
customers_mid_norm = cluster(d[d.season==3],'loadnorm','ID',3,'customer_mid_norm')

# same but with load
customers = cluster(d,'load','ID',4,'customer')
cluster_share(customers)
customers_winter = cluster( d[d.season==1],'load','ID',3,'customer_winter')
cluster_share(customers_winter)
customers_summer = cluster( d[d.season==2],'load','ID',2,'customer_summer')
cluster_share(customers_summer)
customers_mid = cluster(d[d.season==2],'load','ID',2,'customer_mid')
cluster_share(customers_mid)


custo = d.groupby(['Hour','customer']).mean().reset_index()
consumption = d.groupby(['customer']).mean().reset_index()
consumption['consumption_kWh'] = consumption.load*365*24
consumption[['customer','consumption_kWh']]

sns.lineplot(x='Hour', y='load', hue='customer', data=custo)


#### select best ID
best = d
count = best[['ID','Month','load','solar','Year']].groupby(['ID','Month','Year']).count().reset_index()
count = count[count.load>=24*28]
count_month = count.groupby(['ID']).count().reset_index()
count_month = count_month[count_month.Month>=12]
best_ID = count_month.ID.unique()
len(best_ID)
best = best[best.ID.isin(best_ID)]

# look at max
daily_max = best.groupby(['Hour', 'ID']).max().reset_index()
sns.lineplot(x='Hour', y='load', hue='ID', data=daily_max, legend=False)


#
daily = best.groupby(['ID','Hour']).mean().reset_index()
plt.figure(2)
sns.lineplot(x='Hour', y='load', hue='ID', data=daily, estimator=None, legend=False)


customers = cluster(best,'load','ID',4,'customer')


group1_id = customers[customers.clusters==1].ID
group2_id = customers[customers.clusters==2].ID

best_1 = best[best.ID.isin(group1_id)]
daily1 = best_1.groupby(['ID','Hour']).mean().reset_index()
matrix1 = daily1.pivot_table(index=['Hour'], columns=['ID'], values=['load'])
sns.lineplot(x='Hour', y='load', hue='ID', data=daily1)
sns.lineplot(x='Hour', y='load', data=daily1, color='orange')
# Get the average of the group 2 as the tipical profile
load1 = best_1.groupby(['Date']).mean().reset_index()
load1 = load1[(load1.Date >= "2017-07-01 00:00:00") & (load1.Date < "2018-07-01 00:00:00")]
sns.lineplot(x='Date',y='load', data=load1)
load1.to_csv('Load1.csv')
count = load1[['Month','load','solar','Year']].groupby(['Month','Year']).count().reset_index()

best_2 = best[best.ID.isin(group2_id)]
daily2 = best_2.groupby(['ID','Hour']).mean().reset_index()
matrix2 = daily2.pivot_table(index=['Hour'], columns=['ID'], values=['load'])
sns.lineplot(x='Hour', y='load', hue='ID', data=daily2, legend=False).set_title("Customer group 2")
sns.lineplot(x='Hour', y='load', data=daily2, color='cornflowerblue', legend=False)
# Get the average of the group 2 as the tipical profile
load2 = best_2.groupby(['Date']).mean().reset_index()
load2= load2[(load2.Date >= "2017-07-01 00:00:00") & (load2.Date < "2018-07-01 00:00:00")]
sns.lineplot(x='Date',y='load', data=load2)
load2.to_csv('C:/Users\ctouati\PycharmProjects\EquinoxStorage-\Load2.csv')
count = load2[['Month','load','solar','Year']].groupby(['Month','Year']).count().reset_index()



daily2_mean = best_2.groupby(['Hour']).mean().reset_index()
matrix2.to_csv('group2.csv')
daily2_mean[['load']].to_csv('mean.csv')

# graph with all loads
l0 = pd.read_csv('C:/Users\ctouati\PycharmProjects\EquinoxStorage-\Residata_all3.csv')

l1 = pd.read_csv('C:/Users\ctouati\PycharmProjects\EquinoxStorage-\Load1.csv')
l1 = l1[['date','month','day','hour','load']]
l2 = pd.read_csv('C:/Users\ctouati\PycharmProjects\EquinoxStorage-\Load2.csv')
l2 = l2[['date','month','day','hour','load']]

# plot
plt.figure(1)
sns.lineplot(x='hour', y='load', data=l0, color='indianred')
sns.lineplot(x='hour', y='load', data=l1, color='lightblue')
sns.lineplot(x='hour', y='load', data=l2, color='gray')
plt.figure(3)
sns.lineplot(x='hour', y='load', hue='day', data=l0[l0.date <"2017-01-09 00:00:00"], color='indianred', estimator=None)

# yearly consumption
l0.load.sum()*0.75
l1.load.sum()*0.75
l2.load.sum()*0.75

l0['solar_4_180'].sum()
l0['solar_5_180'].sum()
l0['solar_6_180'].sum()
l0['solar_7_180'].sum()