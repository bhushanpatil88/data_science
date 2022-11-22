# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:16.502014Z","iopub.execute_input":"2022-11-22T17:20:16.502493Z","iopub.status.idle":"2022-11-22T17:20:16.509098Z","shell.execute_reply.started":"2022-11-22T17:20:16.502456Z","shell.execute_reply":"2022-11-22T17:20:16.507911Z"}}
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn.pipeline import Pipeline,make_pipeline

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:17.304059Z","iopub.execute_input":"2022-11-22T17:20:17.304524Z","iopub.status.idle":"2022-11-22T17:20:17.349430Z","shell.execute_reply.started":"2022-11-22T17:20:17.304488Z","shell.execute_reply":"2022-11-22T17:20:17.348007Z"}}
df = pd.read_csv("../input/heart-disease-dataset/heart.csv")
df.head()
df.shape

# %% [code] {"jupyter":{"outputs_hidden":false},"_kg_hide-input":false,"execution":{"iopub.status.busy":"2022-11-22T17:20:18.048903Z","iopub.execute_input":"2022-11-22T17:20:18.049355Z","iopub.status.idle":"2022-11-22T17:20:18.289058Z","shell.execute_reply.started":"2022-11-22T17:20:18.049316Z","shell.execute_reply":"2022-11-22T17:20:18.287816Z"}}
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:18.737110Z","iopub.execute_input":"2022-11-22T17:20:18.737563Z","iopub.status.idle":"2022-11-22T17:20:18.797980Z","shell.execute_reply.started":"2022-11-22T17:20:18.737523Z","shell.execute_reply":"2022-11-22T17:20:18.796741Z"}}
df.describe()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:20.532812Z","iopub.execute_input":"2022-11-22T17:20:20.533585Z","iopub.status.idle":"2022-11-22T17:20:20.544276Z","shell.execute_reply.started":"2022-11-22T17:20:20.533540Z","shell.execute_reply":"2022-11-22T17:20:20.542253Z"}}
categorical_cols = []
categorical_inds = []
counting_cols = []
counting_inds  = []
cnt=0
for i in df.columns:
    cnt+=1
    if df[i].nunique()<=5:
        categorical_cols.append(i)
        categorical_inds.append(cnt-1)
    

      
    else:
        counting_cols.append(i)
        counting_inds.append(cnt-1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:21.693204Z","iopub.execute_input":"2022-11-22T17:20:21.693635Z","iopub.status.idle":"2022-11-22T17:20:21.699409Z","shell.execute_reply.started":"2022-11-22T17:20:21.693598Z","shell.execute_reply":"2022-11-22T17:20:21.698155Z"}}
print(categorical_cols)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:22.676984Z","iopub.execute_input":"2022-11-22T17:20:22.677482Z","iopub.status.idle":"2022-11-22T17:20:22.685008Z","shell.execute_reply.started":"2022-11-22T17:20:22.677440Z","shell.execute_reply":"2022-11-22T17:20:22.683500Z"}}
#They are already encoded

categorical_cols.remove('target')
categorical_cols.remove('sex')
categorical_inds.remove(1)
categorical_inds.remove(13)


print(categorical_cols)
print(counting_cols)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:23.856849Z","iopub.execute_input":"2022-11-22T17:20:23.857277Z","iopub.status.idle":"2022-11-22T17:20:23.871250Z","shell.execute_reply.started":"2022-11-22T17:20:23.857241Z","shell.execute_reply":"2022-11-22T17:20:23.869890Z"}}
for i in categorical_cols:
    print(i)
    print(df[i].value_counts())
    print()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:26.363820Z","iopub.execute_input":"2022-11-22T17:20:26.364236Z","iopub.status.idle":"2022-11-22T17:20:28.267036Z","shell.execute_reply.started":"2022-11-22T17:20:26.364202Z","shell.execute_reply":"2022-11-22T17:20:28.265740Z"}}
x = 1
plt.figure(figsize = (20,20))

for i in df.columns:
    plt.subplot(4,4,x)
    plt.boxplot(df[i])
    plt.title(i)
    x = x+1

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:32.435894Z","iopub.execute_input":"2022-11-22T17:20:32.436304Z","iopub.status.idle":"2022-11-22T17:20:32.449365Z","shell.execute_reply.started":"2022-11-22T17:20:32.436272Z","shell.execute_reply":"2022-11-22T17:20:32.447667Z"}}
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(df.drop(['target'],axis=1),
                                                df['target'],
                                                test_size=0.3,
                                                random_state=31)
X_train = X_train.values
y_train = y_train.values
X_test  = X_test.values
y_test  = y_test.values

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:34.037786Z","iopub.execute_input":"2022-11-22T17:20:34.038828Z","iopub.status.idle":"2022-11-22T17:20:34.048255Z","shell.execute_reply.started":"2022-11-22T17:20:34.038776Z","shell.execute_reply":"2022-11-22T17:20:34.047070Z"}}
df.iloc[102,:-1].values

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:35.795020Z","iopub.execute_input":"2022-11-22T17:20:35.795466Z","iopub.status.idle":"2022-11-22T17:20:35.802399Z","shell.execute_reply.started":"2022-11-22T17:20:35.795420Z","shell.execute_reply":"2022-11-22T17:20:35.800740Z"}}
from sklearn.preprocessing import OneHotEncoder
trf3 = make_column_transformer((OneHotEncoder(drop=("first"),handle_unknown='ignore',sparse=False),categorical_inds), remainder='passthrough')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:36.821580Z","iopub.execute_input":"2022-11-22T17:20:36.821996Z","iopub.status.idle":"2022-11-22T17:20:36.828294Z","shell.execute_reply.started":"2022-11-22T17:20:36.821960Z","shell.execute_reply":"2022-11-22T17:20:36.827233Z"}}
from sklearn.preprocessing import RobustScaler
trf4 = make_column_transformer((RobustScaler(),counting_inds),remainder='passthrough')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:39.867715Z","iopub.execute_input":"2022-11-22T17:20:39.868116Z","iopub.status.idle":"2022-11-22T17:20:39.886042Z","shell.execute_reply.started":"2022-11-22T17:20:39.868083Z","shell.execute_reply":"2022-11-22T17:20:39.884736Z"}}
pipe = make_pipeline(trf3,trf4)


X_train = pipe.fit_transform(X_train)
X_test = pipe.transform(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:41.200208Z","iopub.execute_input":"2022-11-22T17:20:41.201022Z","iopub.status.idle":"2022-11-22T17:20:41.210175Z","shell.execute_reply.started":"2022-11-22T17:20:41.200968Z","shell.execute_reply":"2022-11-22T17:20:41.208398Z"}}
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix,accuracy_score


def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print("Test_Set")
    print("confussion matrix")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print("confussion matrix")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))
    print()
    dt_acc_score = accuracy_score(y_test, y_pred)
    print("Accuracy :",dt_acc_score*100,'\n')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:43.144333Z","iopub.execute_input":"2022-11-22T17:20:43.144831Z","iopub.status.idle":"2022-11-22T17:20:43.436760Z","shell.execute_reply.started":"2022-11-22T17:20:43.144789Z","shell.execute_reply":"2022-11-22T17:20:43.435409Z"}}
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(solver='lbfgs', max_iter=10000,random_state=2)
classifier1.fit(X_train, y_train)
y_pred = classifier1.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:50.465501Z","iopub.execute_input":"2022-11-22T17:20:50.466615Z","iopub.status.idle":"2022-11-22T17:20:50.493176Z","shell.execute_reply.started":"2022-11-22T17:20:50.466565Z","shell.execute_reply":"2022-11-22T17:20:50.491915Z"}}
eval_metric(classifier1, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:53.027554Z","iopub.execute_input":"2022-11-22T17:20:53.027992Z","iopub.status.idle":"2022-11-22T17:20:53.037104Z","shell.execute_reply.started":"2022-11-22T17:20:53.027954Z","shell.execute_reply":"2022-11-22T17:20:53.035941Z"}}
from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 2)
classifier2.fit(X_train, y_train)
y_pred = classifier2.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:55.028888Z","iopub.execute_input":"2022-11-22T17:20:55.029575Z","iopub.status.idle":"2022-11-22T17:20:55.048073Z","shell.execute_reply.started":"2022-11-22T17:20:55.029537Z","shell.execute_reply":"2022-11-22T17:20:55.046694Z"}}
eval_metric(classifier2, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:20:56.609741Z","iopub.execute_input":"2022-11-22T17:20:56.610202Z","iopub.status.idle":"2022-11-22T17:21:00.424334Z","shell.execute_reply.started":"2022-11-22T17:20:56.610164Z","shell.execute_reply":"2022-11-22T17:21:00.423248Z"}}
from sklearn.model_selection import GridSearchCV
params = {
    'max_depth': [3,4,5,6,7,8,9,10,11,12,13],
    'min_samples_leaf': [10,20, 25, 75, 50, 100,150,200],
    'criterion': ["gini", "entropy"],
    'splitter':['best','random'],
}          
grid_search = GridSearchCV(estimator = classifier2,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_

print(best_accuracy)
print(best_parameters)
print(best_model)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:21:00.426496Z","iopub.execute_input":"2022-11-22T17:21:00.426965Z","iopub.status.idle":"2022-11-22T17:21:00.633537Z","shell.execute_reply.started":"2022-11-22T17:21:00.426926Z","shell.execute_reply":"2022-11-22T17:21:00.632037Z"}}
from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier()
classifier3.fit(X_train, y_train)
y_pred = classifier3.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:21:00.634878Z","iopub.execute_input":"2022-11-22T17:21:00.635245Z","iopub.status.idle":"2022-11-22T17:21:00.701091Z","shell.execute_reply.started":"2022-11-22T17:21:00.635214Z","shell.execute_reply":"2022-11-22T17:21:00.699902Z"}}
eval_metric(classifier3, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false}}
from sklearn.model_selection import GridSearchCV
params = {
    'n_estimators':[90,100,120,130],
    'max_depth': [3,4,5,6,7,8,9,10,11,12,13],
    'min_samples_leaf': [20, 25, 75],
    'criterion': ["gini", "entropy"],
}          
rf_grid_search = GridSearchCV(estimator = classifier3,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
rf_grid_search.fit(X_train, y_train)
best_accuracy = rf_grid_search.best_score_
best_parameters = rf_grid_search.best_params_
best_model = rf_grid_search.best_estimator_

print(best_accuracy)
print(best_parameters)
print(best_model)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:21:17.973840Z","iopub.execute_input":"2022-11-22T17:21:17.974294Z","iopub.status.idle":"2022-11-22T17:21:18.010039Z","shell.execute_reply.started":"2022-11-22T17:21:17.974253Z","shell.execute_reply":"2022-11-22T17:21:18.008209Z"}}
from sklearn.neighbors import KNeighborsClassifier
classifier4 = KNeighborsClassifier(n_neighbors = 2)
classifier4.fit(X_train, y_train)
y_pred = classifier4.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:21:18.426478Z","iopub.execute_input":"2022-11-22T17:21:18.427192Z","iopub.status.idle":"2022-11-22T17:21:18.519453Z","shell.execute_reply.started":"2022-11-22T17:21:18.427147Z","shell.execute_reply":"2022-11-22T17:21:18.518054Z"}}
eval_metric(classifier4, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:21:18.809815Z","iopub.execute_input":"2022-11-22T17:21:18.810245Z","iopub.status.idle":"2022-11-22T17:21:19.722535Z","shell.execute_reply.started":"2022-11-22T17:21:18.810212Z","shell.execute_reply":"2022-11-22T17:21:19.721157Z"}}
scores=[]

for i in range(1,40):
    classifier4 = KNeighborsClassifier(n_neighbors = i)
    classifier4.fit(X_train, y_train)
    y_pred = classifier4.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:21:21.969682Z","iopub.execute_input":"2022-11-22T17:21:21.970427Z","iopub.status.idle":"2022-11-22T17:21:21.978090Z","shell.execute_reply.started":"2022-11-22T17:21:21.970364Z","shell.execute_reply":"2022-11-22T17:21:21.976707Z"}}
maxi_ind=-1
maxi=0
cnt=0
for i in scores:
    cnt+=1
    if maxi<i:
        max_ind = cnt
        maxi = i
print(max_ind)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:21:23.602524Z","iopub.execute_input":"2022-11-22T17:21:23.602954Z","iopub.status.idle":"2022-11-22T17:21:23.631678Z","shell.execute_reply.started":"2022-11-22T17:21:23.602920Z","shell.execute_reply":"2022-11-22T17:21:23.630337Z"}}
classifier4 = KNeighborsClassifier(n_neighbors = max_ind)
classifier4.fit(X_train, y_train)
y_pred = classifier4.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:21:24.831732Z","iopub.execute_input":"2022-11-22T17:21:24.832765Z","iopub.status.idle":"2022-11-22T17:21:24.909893Z","shell.execute_reply.started":"2022-11-22T17:21:24.832718Z","shell.execute_reply":"2022-11-22T17:21:24.908551Z"}}
eval_metric(classifier4, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:21:27.457733Z","iopub.execute_input":"2022-11-22T17:21:27.458168Z","iopub.status.idle":"2022-11-22T17:21:27.469431Z","shell.execute_reply.started":"2022-11-22T17:21:27.458134Z","shell.execute_reply":"2022-11-22T17:21:27.468310Z"}}
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(df.drop(['target'],axis=1),
                                                df['target'],
                                                test_size=0.3,
                                                random_state=31)

X_train = X_train.values
y_train = y_train.values
X_test  = X_test.values
y_test  = y_test.values

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:23:01.313360Z","iopub.execute_input":"2022-11-22T17:23:01.313861Z","iopub.status.idle":"2022-11-22T17:23:01.320033Z","shell.execute_reply.started":"2022-11-22T17:23:01.313822Z","shell.execute_reply":"2022-11-22T17:23:01.318913Z"}}
from sklearn.preprocessing import OneHotEncoder
trf3 = make_column_transformer((OneHotEncoder(drop=("first"),handle_unknown='ignore',sparse=False),categorical_inds), remainder='passthrough')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:23:01.903422Z","iopub.execute_input":"2022-11-22T17:23:01.904415Z","iopub.status.idle":"2022-11-22T17:23:01.909732Z","shell.execute_reply.started":"2022-11-22T17:23:01.904373Z","shell.execute_reply":"2022-11-22T17:23:01.908510Z"}}
from sklearn.preprocessing import RobustScaler
trf4 = make_column_transformer((RobustScaler(),counting_inds),remainder='passthrough')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:23:36.074837Z","iopub.execute_input":"2022-11-22T17:23:36.075270Z","iopub.status.idle":"2022-11-22T17:23:36.294267Z","shell.execute_reply.started":"2022-11-22T17:23:36.075235Z","shell.execute_reply":"2022-11-22T17:23:36.292976Z"}}
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:23:37.646317Z","iopub.execute_input":"2022-11-22T17:23:37.646786Z","iopub.status.idle":"2022-11-22T17:23:37.858846Z","shell.execute_reply.started":"2022-11-22T17:23:37.646734Z","shell.execute_reply":"2022-11-22T17:23:37.857540Z"}}
pipe = make_pipeline(trf3,trf4,classifier)


pipe.fit(X_train,y_train)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:23:40.246546Z","iopub.execute_input":"2022-11-22T17:23:40.246991Z","iopub.status.idle":"2022-11-22T17:23:40.278396Z","shell.execute_reply.started":"2022-11-22T17:23:40.246958Z","shell.execute_reply":"2022-11-22T17:23:40.277181Z"}}
y_pred = pipe.predict(X_test)
y_pred

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:23:42.008207Z","iopub.execute_input":"2022-11-22T17:23:42.008657Z","iopub.status.idle":"2022-11-22T17:23:42.072970Z","shell.execute_reply.started":"2022-11-22T17:23:42.008606Z","shell.execute_reply":"2022-11-22T17:23:42.071782Z"}}
eval_metric(pipe, X_train, y_train, X_test, y_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:23:44.810560Z","iopub.execute_input":"2022-11-22T17:23:44.811813Z","iopub.status.idle":"2022-11-22T17:23:44.826272Z","shell.execute_reply.started":"2022-11-22T17:23:44.811762Z","shell.execute_reply":"2022-11-22T17:23:44.824958Z"}}
import pickle

pickle.dump(pipe,open('pipe.pkl','wb'))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:23:46.121705Z","iopub.execute_input":"2022-11-22T17:23:46.122153Z","iopub.status.idle":"2022-11-22T17:23:46.136307Z","shell.execute_reply.started":"2022-11-22T17:23:46.122117Z","shell.execute_reply":"2022-11-22T17:23:46.134782Z"}}
pipe = pickle.load(open('pipe.pkl','rb'))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:23:46.600796Z","iopub.execute_input":"2022-11-22T17:23:46.601263Z","iopub.status.idle":"2022-11-22T17:23:46.608465Z","shell.execute_reply.started":"2022-11-22T17:23:46.601222Z","shell.execute_reply":"2022-11-22T17:23:46.606853Z"}}
test1 = np.array([53,1,0,140,203,1,0,155,1,3.1,0,0,3]).reshape(1,13)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:23:47.134327Z","iopub.execute_input":"2022-11-22T17:23:47.135271Z","iopub.status.idle":"2022-11-22T17:23:47.141613Z","shell.execute_reply.started":"2022-11-22T17:23:47.135222Z","shell.execute_reply":"2022-11-22T17:23:47.140398Z"}}
test2 = np.array([45,1,2,255,255,1,2,255,1,1.4,2,4,3]).reshape(1,13)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:23:47.646191Z","iopub.execute_input":"2022-11-22T17:23:47.646674Z","iopub.status.idle":"2022-11-22T17:23:47.672683Z","shell.execute_reply.started":"2022-11-22T17:23:47.646620Z","shell.execute_reply":"2022-11-22T17:23:47.671295Z"}}
pipe.predict(test1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-11-22T17:23:50.574866Z","iopub.execute_input":"2022-11-22T17:23:50.575493Z","iopub.status.idle":"2022-11-22T17:23:50.597943Z","shell.execute_reply.started":"2022-11-22T17:23:50.575445Z","shell.execute_reply":"2022-11-22T17:23:50.597016Z"}}
pipe.predict(df.iloc[102,:-1].values.reshape(1,13))

# %% [code] {"jupyter":{"outputs_hidden":false}}
