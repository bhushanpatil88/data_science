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

# Importing Libraries
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import kerastuner as kt

#Initializing ANN
classifier = Sequential()

#Adding input and hidden layer
classifier.add(Dense(32,activation='relu',input_dim=10))
classifier.add(BatchNormalization())
classifier.add(Dense(16,activation='relu'))
classifier.add(BatchNormalization())
# Dropout Layer
'''
WE add dropout layer to avoid overfitting of data 
which causes the ANN to randomly select few neurons
'''
#classifier.add(Dropout(0.1))

classifier.add(Dense(8,activation='relu'))
classifier.add(BatchNormalization())

# Adding the output layer    
'''
For classification if binary use sigmoid else use softmax
'''        
classifier.add(Dense(1,activation='sigmoid'))

# Get the overall summary of total parameters and connections
print(classifier.summary())

# Compiling the model 
classifier.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

# Fitting model to training data with validation split of 20%
'''
If the batch size is 1 it is  stochastics gradient descent  
if batch size is equal to (number of rows) it is batch gradient descent
else it is mini batch gradient descent 
'''
history = classifier.fit(X_train,y_train,epochs=100,batch_size=256,validation_data=(X_test,y_test)) 

# Getting 1st layer weights and bias
classifier.layers[0].get_weights()

# Making predictins
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#Accuracy
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test, y_pred)
print("\n")

confusion_matrix(y_test, y_pred)

# Plotting the test and train graphs to check overfitting

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


'''
Keras Tuner
'''
def build_mode(hp):
  classifier = Sequential()

  counter = 0

  for i in range(hp.Int('num_layers',min_value=1,max_value=10,step=2)):
    if counter==0:
      classifier.add(Dense(
          hp.Int('units_'+str(i),min_value=8,max_value=128),
          activation = hp.Choice('activation_'+str(i),values=['relu','sigmoid']),
          input_dim=10
          ))
      classifier.add(BatchNormalization())
      classifier.add(Dropout(hp.Choice('drop1'+str(i),values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
      classifier.add(BatchNormalization())
    else:
      classifier.add(Dense(
          hp.Int('units_'+str(i),min_value=8,max_value=128),
          activation = hp.Choice('activation_'+str(i),values=['relu','sigmoid'])
          ))
      classifier.add(BatchNormalization())
      classifier.add(Dropout(hp.Choice('drop1'+str(i),values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
      classifier.add(BatchNormalization())

    counter+=1

  classifier.add(Dense(1,activation='sigmoid'))

  optimizer = hp.Choice('optimizer',values = ['adam','sgd','rmsprop','adadelta'])
  classifier.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
  return classifier

tuner = kt.RandomSearch(build_mode,objective='val_accuracy',max_trials=3,directory='mydir1',project_name='bhushan')
tuner.search(X_train,y_train,epochs=10,validation_data = (X_test,y_test))
tuner.get_best_hyperparameters()[0].values
classifier = tuner.get_best_models(num_models=1)[0]

history = classifier.fit(X_train,y_train,epochs=200,batch_size=128,initial_epoch=11,validation_data=(X_test,y_test))

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#Accuracy
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test, y_pred)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

# Batch_size and epochs

def build():    
    classifier = tuner.get_best_models(num_models=1)[0]
    return classifier

from keras.wrappers.scikit_learn import KerasClassifier

classifier = KerasClassifier(build_fn=build)

from sklearn.model_selection import GridSearchCV
parameters = {
    'batch_size':[256,512],
    'nb_epoch':[100,200],
    }
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
print(best_accuracy)
best_parameters = grid_search.best_params_

history = classifier.fit(X_train,y_train,epochs=300,batch_size=128,initial_epoch=11,validation_data=(X_test,y_test))

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#Accuracy
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test, y_pred)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])



