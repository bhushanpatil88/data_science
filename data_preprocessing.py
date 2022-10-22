import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#To see the pipeline
from sklearn import set_config
set_config(display='diagram')

df = pd.read_csv("titanic_train.csv")


#checking missing values
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


#drop missing values if many
df.drop(['Cabin'],axis=1,inplace=True)


#drop duplicates
df.drop_duplicates(inplace=True)


#Removing string data
df.drop(["Name","Ticket"],axis=1,inplace=True)





#Doning the train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(df.drop(['Survived'],axis=1),
                                                df['Survived'],
                                                test_size=0.3,
                                                random_state=101)



#fill missing values
# Use mean if no outliers
# Use median if outliers
# Use mode for categorial data
from sklearn.impute import SimpleImputer
trf1 = ColumnTransformer(transformers=[
        ('mean',SimpleImputer(),[3]),
       # ('median',SimpleImputer(strategy='median'),[3]),
        #('mode',SimpleImputer(strategy='most_frequent'),[3])
    ],remainder='passthrough')

#Encoding Categorial data -> OrdinalEncoding
# It is used when the categories are dependent like good,bad,very_good
from sklearn.preprocessing import OrdinalEncoder
trf2 = ColumnTransformer(transformers=[
        ('ordinal1',OrdinalEncoder(categories=[['Poor','Average','Good'],['A+','A','B']]),[])
        ],remainder='passthrough')



#Encoding categorial data->OneHotEncoding
# It is used when the categories are independent of each other like male_female
from sklearn.preprocessing import OneHotEncoder
trf3 = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(drop=("first"),handle_unknown='ignore',sparse=False), [3,7])
    ], remainder='passthrough')




#Normalizing the data
from sklearn.preprocessing import StandardScaler
trf4 = ColumnTransformer(transformers=[
    ('scaler',StandardScaler(),slice(0,10))
    ],remainder='passthrough')





#creating pipeline
pipe = Pipeline([
    ('1',trf1),
   ('3',trf3),
    ('4',trf4),
    ])


X_train = pipe.fit_transform(X_train)
X_test = pipe.transform(X_test)

# Encoding the Dependent Variable
# if the dependent variable is categorical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


# Dimension Reduction
# PCA
'''
We use PCA for finding the 
dimensions on the criteria
having most variance between two categories
PC1 > PC2 > PC3... 
'''
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)


# LCA
'''
WE use LCA for maximising 
the separation between two categories
It is mostly similiar to PCA 
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# Post Processing
# GridSearchCV
'''
Use for hyper-parameters tunning to select the best parameter
which will give the highest accuracy
'''
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# K-Fold_Cross_Validation
'''
WE use 10-fold CV to check finding the 
best train test split
'''
from sklearn.model_selection import cross_val_score,KFold
kfold = KFold(10)
def best_model(model):
    scores = cross_val_score(model,X,y,cv=kfold)
    print(scores)
    
