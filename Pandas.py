import pandas as pd
import numpy as np


site_id = 1058
df = pd.read_csv('/Users/coralietouati/PycharmProjects/Project1/' + str(site_id) + '_risk.csv')

df_filtered = df[(df['operating year'] == 1)]
percentile = df_filtered.ess_kW_savings.quantile(0.5)
df_filtered['Delta_percentile'] = df.apply( lambda row: row['ess_kW_savings'] - percentile, axis=1 )


data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'age': [42, 52, 36, 24, 73],
        'preTestScore': [4, 24, 31, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70]}
df2 = pd.DataFrame(data, columns = ['name', 'age', 'preTestScore', 'postTestScore'])


# Affichage des info de la df
    print('df2.shape')
    df.head() # le debut de la base de données
    df.tail() # la fin des données
    print(df.columns)
    # type de chaque colonne #
    print(df.dtypes)
    print(df.info())
    print(df.describe(include='all'))

# Manipulation variable
    print(df['ess_kW_savings'])
    # autre methodes pas attentions aux espaces!
    print(df.ess_kW_savings)
    print(df[['iteration','ess_kW_savings']])

    print(df['ess_kW_savings'].describe())
    print(df['ess_kW_savings'].mean())
    print(df['ess_kW_savings'].value_counts())

    print(df['ess_kW_savings'][1])
    print(df['ess_kW_savings'][1:3])

    # Trier les donnees
    print(df2.age)
    print(df2['age'].sort_values())
    print(df2['age'].argsort()) # donne l'index des valeurs triées
    print(df2.sort_values(by='age').head())

# Itérations sur les variables
    for col in df2.columns:
        print(df2[col].dtype)

#Accès indicé aux données d'un DataFrame
    # iloc permet d'utiliser les indices
    print(df2.iloc[0,0])
    print(df2['name'][0])
    print(df2)
    #valeur située en dernière ligne, première colonne --> utilisation de l'indiçage négatif
    print(df2.iloc[-1,0])
    print(df2.shape[0]) # nb de ligne
    print(df2.shape[1]) # nb de colonne
    print(df2.iloc[df.shape[0]-1,0]) # equivalent à la méthode -1
    print(df2.iloc[0:5,:]) # ligne 0 à 5 et toutes les colonnes
    print(df2.iloc[-2:,:])
    print(df2.loc[-2:,:])
    print(df2.iloc[0:5,[0,2]])
    print(df2.iloc[0:5,0:3:2]) # idem
    # difference between loc and iloc:
        #loc gets rows (or columns) with particular labels from the index.
    print(df2.iloc[0:3,0])
        #iloc gets rows (or columns) at particular positions in the index (so it only takes integers)


#FILTRER
    print(df2[( df2['age'] > 30 )])
    print( df2.loc[ df2['age'] > 30 , :])
    print( df2['age'] > 30 )
    print( (df2['age'] > 30).value_counts())

    colonnes = ['name','age']
    print(df2.loc[ ( (df2.age > 30) & (df2.preTestScore >10) ), colonnes])

    # filter a partir dune liste
d[d.ID.isin(ID_keep)]

    # change value of some rows only
d.loc[ (d['week'] == 52) & (d['month'] == 1 ), 'week'] = 0

    #groupby
d[d['week']==0].groupby(['day']).mean()

    # fin the max of a column for a given row
min(d.loc[1:200,'day'])



#Calculs récapitulatifs - Croisement des variables
    # Tableaux croisé dynamique
    print(pd.crosstab( df2.age , df2.preTestScore))

    df2['Result'] = np.where( df2.preTestScore > 10 , 'Valid', 'Fail') # voir np.select si plusieurs conditions

    # add column: We can use DataFrame.apply to apply a function to all columns axis=0 (the default) or axis=1 rows.
    df2['Total'] = df2.preTestScore.values + df2.postTestScore.values
    type(df2.preTestScore)
    # Ajout colonne
    df2.loc[:, 'Tot'] = df2.preTestScore.values + df2.postTestScore.values
    #Inverser la valeurs de deux colonnes
    df2.loc[:,['Tot', 'preTestScore']] = df2[['preTestScore', 'Tot']].values
    df2['Total'] = 0

    print(df2)

# Boucle for avec condition
for i in range(0,int(df2.shape[0])):
    if (df2.loc[i, 'postTestScore']-df2.loc[i, 'age']) >=0:
     df2.loc[i, 'test'] = 1
    else:
        df2.loc[i, 'test'] = df2.loc[i, 'postTestScore']/df2.loc[i, 'age']

# Sommer une colonne
    print(df2['preTestScore'].sum() )
    # avec une condition, sum toute les colonnes, on peut choisir a la fin
    print(df2[ df2['preTestScore'] > 10 ].sum())
    print(df2[df2['preTestScore'] > 10].sum())[2]
    print(df2.preTestScore[ df2['preTestScore'] > 10 ].sum())
    print(df2['preTestScore'][ df2['preTestScore'] > 10 ].sum())
# Sumprod
    sum(np.multiply(df2['age'],df2['age']))

# test
for i in df2.age:
        print(i)

# Supprimer les colonnes NAN
    d.dropna
#supprimer les lignes qui contiennent une chaine de caractaire
load = load[load['local_dt'].str.contains('2016-02-29') == False]

# open a database from an excel
pv = pd.read_excel("file.xlsx", sheetname = 'Summary of Results', skiprows=107)

# dowload
df.to_csv('excel.csv')

# Renomer un file
# import os
os.rename('solar1.csv', str(site_id) + '_solar.csv')


# dealing with nan values
d.isnull()

# Give value with condition without boucle for
d['Week']= np.where(d.Day>=5,'Weekend','Week')

# better way of giving a vlue bassed on other columns value
d.loc[d.ID == id, 'max'] = 'maxval'