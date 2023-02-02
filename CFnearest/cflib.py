from scipy.spatial import cKDTree
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import xgboost
import sklearn
from aif360.sklearn.datasets import fetch_adult,fetch_bank,fetch_compas,fetch_german
from itertools import repeat
from sklearn.preprocessing import LabelEncoder

dataset = ['adult' , 'compas' ,'german']

def load_dataset(dataset):
    if dataset == 'adult':
        X, y, _ = fetch_adult()
    elif dataset == 'compas':
        X,y = fetch_compas()
    else :
        X,y = fetch_german()
    return X,y

def create_df(X,y,dataset):
    df = X.copy()
    df['label'] = y
    if dataset == 'adult':
        df = df.drop(columns='education',axis=1)
        #df = df.drop(columns=['sex','race'])
        df = df.reset_index(drop=True)
    elif dataset == 'compas':
        df = df.drop(columns=['sex','race'])
        df = df.reset_index()
    else: 
        df = df.drop(columns=['sex','age','foreign_worker'])
        df = df.reset_index()
    return df

def preprocess(df,dataset):
    if dataset == 'adult':
        #df['marital-status'].replace('Never-married', 'NotMarried',inplace=True)
        #df['marital-status'].replace(['Married-AF-spouse'], 'Married',inplace=True)
        #df['marital-status'].replace(['Married-civ-spouse'], 'Married',inplace=True)
        #df['marital-status'].replace(['Married-spouse-absent'], 'NotMarried',inplace=True)
        #df['marital-status'].replace(['Separated'], 'Separated',inplace=True)
        #df['marital-status'].replace(['Divorced'], 'Separated',inplace=True)
        #df['marital-status'].replace(['Widowed'], 'Widowed',inplace=True)
        df['label'] = df['label'].replace(['<=50K'], 0)
        df['label'] = df['label'].replace(['>50K'], 1)
        df.label = pd.to_numeric(df.label)
        #df['race'] = df['race'].replace(['Non-white'], 0)
        #df['race'] = df['race'].replace(['White'], 1)
        #df.race = pd.to_numeric(df.race)
        df['sex'] = df['sex'].replace(['Female'], 0)
        df['sex'] = df['sex'].replace(['Male'], 1)
        df.sex = pd.to_numeric(df.sex)
        #df.age = pd.qcut(df.age,q=3)
        #df['hours-per-week'] = pd.cut(df['hours-per-week'],bins=[0,40,50,99])
        #df['capital-loss'] = pd.cut(df['capital-loss'],bins=[-0.001,1000,2000,4500])
        #df['capital-gain'] = pd.cut(df['capital-gain'],bins=[-0.001,10000,50000,99999])
        #df = pd.get_dummies(df)
        labelencoder = LabelEncoder()
        categ = df.select_dtypes(include = "category").columns
        for feat in categ:
            df[feat] = labelencoder.fit_transform(df[feat])


    elif dataset == 'compas':
        df['label'] = df['label'].replace(['Recidivated'], 0)
        df['label'] = df['label'].replace(['Survived'], 1)
        df.label = pd.to_numeric(df.label)
        df['race'] = df['race'].replace(['African-American'], 0)
        df['race'] = df['race'].replace(['Caucasian'], 1)
        df.race = pd.to_numeric(df.race)
        df['sex'] = df['sex'].replace(['Male'], 0)
        df['sex'] = df['sex'].replace(['Female'], 1)   
        df.sex = pd.to_numeric(df.sex)
        df = df.drop(columns = 'c_charge_desc', axis=1)   
        df = df.drop(columns = 'age', axis=1) 
        df['juv_fel_count'] = np.where(df['juv_fel_count'] !=  0, 'Yes','No') 
        df['juv_misd_count'] = np.where(df['juv_misd_count'] !=  0, 'Yes','No')
        df['juv_other_count'] = np.where(df['juv_other_count'] !=  0, 'Yes','No')
        df['priors_count'] = pd.cut(df['priors_count'],bins=[-0.01,5,10,38])
        df = pd.get_dummies(df)

    else :
        df['sex'] = df['sex'].replace(['male'], 1)
        df['sex'] = df['sex'].replace(['female'], 0)   
        df.sex = pd.to_numeric(df.sex)
        df['age'] = df['age'].replace(['aged'], 1)
        df['age'] = df['age'].replace(['young'], 0)  
        df.age = pd.to_numeric(df.age)
        df['label'] = df['label'].replace(['bad'], 0)
        df['label'] = df['label'].replace(['good'], 1)
        df.label = pd.to_numeric(df.label)

        df.duration = pd.qcut(df.duration,q=3)
        df.credit_amount = pd.qcut(df.credit_amount,q=3)
        df = pd.get_dummies(df)     
        
    return df

def percentage(values,y):
    return round((values/y)*100,2)

def most_occured(df):
    lists = df.columns
    
    mostly_occurred_items = []
    values = []
    df1 =[]
    i = 0
     
    for items in lists:
        value_counts =  df[items].value_counts()
        
        for index,item in value_counts.items():            
            mostly_occurred_items.append(index)
            values.append(item)
            break
            
        values = list(map(percentage,values,repeat(df.shape[0])))
        df1.append({
                "Data":items,
                "Most occurred Category":mostly_occurred_items[i],
                "Percentage (%)":values[i]})
        i += 1

    return pd.DataFrame(df1,columns=["Data","Most occurred Category","Percentage (%)"])

def group(df,sensitive_attribute):
    priv=df.loc[df[sensitive_attribute] == 1]
    unpriv=df.loc[df[sensitive_attribute] == 0]

    affected_priv=priv.loc[df['label'] == 0]
    affected_unpriv=unpriv.loc[df['label'] == 0]
    unaffected_priv=priv.loc[df['label'] == 1]
    unaffected_unpriv=unpriv.loc[df['label'] == 1]

    a_priv=affected_priv.drop([sensitive_attribute,"label"], axis=1)
    a_priv = a_priv.reset_index()
    a_priv = a_priv.drop(columns = 'index')
    a_unpriv=affected_unpriv.drop([sensitive_attribute,"label"], axis=1)
    a_unpriv = a_unpriv.reset_index()
    a_unpriv = a_unpriv.drop(columns = 'index')
    u_priv=unaffected_priv.drop([sensitive_attribute,"label"], axis=1)
    u_priv = u_priv.reset_index()
    u_priv = u_priv.drop(columns = 'index')
    u_unpriv=unaffected_unpriv.drop([sensitive_attribute,"label"], axis=1)
    u_unpriv = u_unpriv.reset_index()
    u_unpriv = u_unpriv.drop(columns = 'index')

    return a_priv,a_unpriv,u_priv,u_unpriv

def nearest(affected, kdtree, unaffected):
    affected_priv=np.asarray(affected)
    unaffected_priv=np.asarray(unaffected)
    match = []
    result =[]
    index = []


    for item in affected_priv: #affected
        matching=kdtree.query(item,k=1,p=2) #unaffected
        result.append([item,unaffected_priv[matching[1]], matching[0]])
        match.append(matching[0])
        index.append(matching[1])

    return match,result,index

def KDTree(unaffected):
    unaffected_np=np.asarray(unaffected)
    un_tree = cKDTree(unaffected_np, leafsize=10)
    return un_tree

from sklearn.neighbors import NearestNeighbors
def nn_closest(df):
    closest = df.to_numpy()
    nbrs = NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(closest)
    distances, indices = nbrs.kneighbors(closest)

    indexes = []
    for i in range(len(indices)):
        indexes.append(indices[i][1])

    df['indexes'] = indexes

    distance = []

    for i in range(len(distances)):
        distance.append(distances[i][1])
    
    df['distances'] = distance

    return df
