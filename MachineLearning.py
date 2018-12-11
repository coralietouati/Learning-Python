import os
import tarfile
from six.moves import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, cross_val_predict
from sklearn.preprocessing import Imputer, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier


Download_root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
Housing_path = os.path.join("datasets","housing")
Housing_url = Download_root + "datasets/housing/housing.tgz"

def fetch_housing__data(housing_url = Housing_url, housing_path = Housing_path):
    if not os.path.isdir(housing_path): # if the file doesnt exist
        os.makedirs(housing_path)
        print('creating the file datasets/housing')
    tgz_path = os.path.join(housing_path, "housing.tar")
    urllib.request.urlretrieve(housing_url,tgz_path) # download the zip file
    housing_tgz =  tarfile.open(tgz_path) # go in the right directory
    housing_tgz.extractall(path=housing_path) # and extract the file in the new file
    housing_tgz.close()


def load_housing_data(housing_path = Housing_path):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)



######################## DATA ANALYSIS #####################################

housing = load_housing_data()

## QUICK LOOK

housing.head()
housing.info()
housing.ocean_proximity.value_counts()
housing.describe()

housing.hist(bins=50)
plt.show()


## CREATE A TEST SET

    #option 1 simple: mais a chaque fois different, a force l'algo ne sera entrainer sur tout le dataset
def split_train_test(data, ratio):
    suffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*ratio)
    test_indices = suffled_indices[:test_set_size]
    train_indices = suffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)

    #option 2: complexe ---> hash ??
def test_set_check(identifier, ratio, hash):
    return hash(np.int64(identifier)).digest()[-1]< 256 * ratio # transform id in array

def split_train_test_by_id(data, ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, ratio, hash))
    print(in_test_set)
    return data.loc[~in_test_set], data[in_test_set]

housing = housing.reset_index() # to add the index as the id column
train_set, test_set = split_train_test_by_id(housing, 0.2, "index")
print(str(len(train_set)) + " in the train set and " + str(len(test_set)) + " in the test set")

    # option 3 avec sklearn
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)



# create a income category to check
housing['income_cat'] = np.ceil(housing.median_income/1.5)
housing.income_cat.unique()
housing.income_cat.where(housing.income_cat < 5, 5, inplace=True) # bizareeeee
housing.income_cat.unique()
housing.income_cat.hist()
plt.show()

# option 2 avec sklearn
split =  StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing.income_cat):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

#remove the category to back to normal
for set in (strat_test_set, strat_test_set):
    print(set)
    set.drop("income_cat", axis=1, inplace=True)


## DATA VIZUALISATION

housing = test_set.copy()
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing.population/100, label='population',
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)


## CORRELATION
corr_matrix = housing.corr()
corr_matrix.median_house_value.sort_values(ascending=False)

attributes = ['median_house_value', 'mediam_income', 'total_rooms', 'housing_mediam_age']
scatter_matrix(housing[attributes], figsize =(2,8))
housing.plot(kind='scatter', x = 'median_income', y='median house value')

# combinaison d'attributes
housing['rooms_per_household'] = housing.total_rooms/housing.households
corr_matrix = housing.corr()
corr_matrix.median_house_value.sort_values(ascending=False)


## PREP FOR ML

housing = strat_train_set.drop('median_house_value', axis=1)
housing_label = strat_train_set['median_house_value'].copy()

#Data cleaning
# 3 way todeal with missing values
housing.dropna(subset=["total_bedrooms"]) # get rid of lines with missing values
housing.drop("total_bedrooms", axis=1) # get rif of the attribute
mean = housing.total_bedrooms.mean()
housing.total_bedrooms.fillna(mean, inplace=True) # replace with median
# same with sk learn
housing_num = housing.drop('ocean_proximity', axis=1)
imputer =Imputer(strategy="median")
imputer.fit(housing_num)# Imputer can only be run on numerical dataframe
imputer.statistics_
housing_num.median().values
X = imputer.transform(housing_num) # numpy
housing_tr= pd.DataFrame(X, columns=housing_num.columns)

# Text category
housing_cat = housing.ocean_proximity
encoder = OneHotEncoder()
housing_cat_encoded, housing_categories = housing_cat.factorize() # convert categories into integers
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1)) # -1 is the length of the 1D dimension given

binarycat = pd.DataFrame(housing_cat_1hot.toarray(), columns=housing_categories)
housing_notext = pd.concat([housing.drop('ocean_proximity', axis=1).reset_index(drop=True), binarycat], axis=1)
housing = housing_notext

# Custom Transformers
room_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """Herite des 2 classes de sklearn"""
    def __init__(self, add_bedrooms_per_room = True):  # no *argpr **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, room_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix]/ X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room= X[:, bedrooms_ix ] / X[:, room_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# Feature scaling of the training data

# Normalization: (x-Min)/max
#fit_transform(X[, y])	Fit to data, then transform it.
scaler = MinMaxScaler()
housing_norm = scaler.fit_transform(housing)
housing_norm = pd.DataFrame(housing_norm, columns=housing.columns)
print(scaler.data_max_)
print(scaler.data_min_)
housing_norm.max()

# Standardize: (X-mean)/variance
scaler = StandardScaler()
housing_norm = scaler.fit_transform(housing)
housing_norm = pd.DataFrame(housing_norm, columns=housing.columns)
housing_norm.max() # not working with the binary variable

scatter_matrix(housing_num)
plt.style.use('seaborn-white')
housing_norm.hist(bins=np.arange(0,1.2,0.2), ec='w')


# Tranformation Pipelines

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


##################### SUM UP  ###############
housing = load_housing_data()
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
housing = train_set.drop('median_house_value', axis=1)
housing_label = train_set['median_house_value'].copy()
housing_num = housing.drop('ocean_proximity', axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]


num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy='median')),
    ('attributs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])


# # autre facond e faire un pipeline
# model = make_pipeline(Imputer(strategy='mean'),
#                       PolynomialFeatures(degree=2),
#                       LinearRegression())

housing_prepared = full_pipeline.fit_transform(housing)

col = list(housing.columns) + list(housing.ocean_proximity.unique())
housing_prepared_pd = pd.DataFrame(housing_prepared.toarray(), columns=col)


### SELECT AND TRAIN A MODEL

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_label)


housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_label, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse) # underfitting!

# 2eme model: Decision Tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_label)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_label, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse) # overfitting !


# Cross Validation
#K-fold cross-validation
scores = cross_val_score(tree_reg, housing_prepared, housing_label, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores) # utility function and not cost


# Save your models
joblib.dump(tree_reg, "treemodel.pkl")
model = joblib.load("treemodel.pkl")


# See where the model is wrong
mat = confusion_matrix(housing)

## FINE TUNE YOUR MODELS
# Grid search

## Once you selected the ifnal model, Evaluate model en test set
X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)  # AND NOT FIT !!
final_predictions = lin_reg.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)