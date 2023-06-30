# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:00:07.675564Z","iopub.execute_input":"2023-06-29T19:00:07.675961Z","iopub.status.idle":"2023-06-29T19:00:07.681087Z","shell.execute_reply.started":"2023-06-29T19:00:07.675933Z","shell.execute_reply":"2023-06-29T19:00:07.679921Z"}}
import pandas as pd

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:03:44.876534Z","iopub.execute_input":"2023-06-29T19:03:44.876973Z","iopub.status.idle":"2023-06-29T19:03:44.886959Z","shell.execute_reply.started":"2023-06-29T19:03:44.876944Z","shell.execute_reply":"2023-06-29T19:03:44.886003Z"}}
data = pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:04:26.574293Z","iopub.execute_input":"2023-06-29T19:04:26.575205Z","iopub.status.idle":"2023-06-29T19:04:26.615510Z","shell.execute_reply.started":"2023-06-29T19:04:26.575152Z","shell.execute_reply":"2023-06-29T19:04:26.614442Z"}}
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:06:27.003862Z","iopub.execute_input":"2023-06-29T19:06:27.004307Z","iopub.status.idle":"2023-06-29T19:06:27.027329Z","shell.execute_reply.started":"2023-06-29T19:06:27.004272Z","shell.execute_reply":"2023-06-29T19:06:27.026220Z"}}
data.tail()

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:06:41.346153Z","iopub.execute_input":"2023-06-29T19:06:41.346527Z","iopub.status.idle":"2023-06-29T19:06:41.353796Z","shell.execute_reply.started":"2023-06-29T19:06:41.346499Z","shell.execute_reply":"2023-06-29T19:06:41.352628Z"}}
data.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:06:54.930676Z","iopub.execute_input":"2023-06-29T19:06:54.931881Z","iopub.status.idle":"2023-06-29T19:06:54.938358Z","shell.execute_reply.started":"2023-06-29T19:06:54.931840Z","shell.execute_reply":"2023-06-29T19:06:54.937133Z"}}
print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:07:50.554542Z","iopub.execute_input":"2023-06-29T19:07:50.555688Z","iopub.status.idle":"2023-06-29T19:07:50.570597Z","shell.execute_reply.started":"2023-06-29T19:07:50.555645Z","shell.execute_reply":"2023-06-29T19:07:50.569482Z"}}
data.info()

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:08:40.557138Z","iopub.execute_input":"2023-06-29T19:08:40.557632Z","iopub.status.idle":"2023-06-29T19:08:40.570641Z","shell.execute_reply.started":"2023-06-29T19:08:40.557581Z","shell.execute_reply":"2023-06-29T19:08:40.569396Z"}}
data.isnull().sum()

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:08:45.723170Z","iopub.execute_input":"2023-06-29T19:08:45.723576Z","iopub.status.idle":"2023-06-29T19:08:45.735656Z","shell.execute_reply.started":"2023-06-29T19:08:45.723543Z","shell.execute_reply":"2023-06-29T19:08:45.734596Z"}}
data.isnull().sum()*100 / len(data)

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:09:45.455197Z","iopub.execute_input":"2023-06-29T19:09:45.455628Z","iopub.status.idle":"2023-06-29T19:09:45.471897Z","shell.execute_reply.started":"2023-06-29T19:09:45.455584Z","shell.execute_reply":"2023-06-29T19:09:45.470859Z"}}
data.head(1)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:10:11.282702Z","iopub.execute_input":"2023-06-29T19:10:11.283409Z","iopub.status.idle":"2023-06-29T19:10:11.288684Z","shell.execute_reply.started":"2023-06-29T19:10:11.283365Z","shell.execute_reply":"2023-06-29T19:10:11.287519Z"}}
columns = ['Gender','Dependents','LoanAmount','Loan_Amount_Term']

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:10:33.917234Z","iopub.execute_input":"2023-06-29T19:10:33.918006Z","iopub.status.idle":"2023-06-29T19:10:33.926065Z","shell.execute_reply.started":"2023-06-29T19:10:33.917965Z","shell.execute_reply":"2023-06-29T19:10:33.924897Z"}}
data = data.dropna(subset=columns)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:10:47.067968Z","iopub.execute_input":"2023-06-29T19:10:47.068333Z","iopub.status.idle":"2023-06-29T19:10:47.080861Z","shell.execute_reply.started":"2023-06-29T19:10:47.068307Z","shell.execute_reply":"2023-06-29T19:10:47.079772Z"}}
data.isnull().sum()*100 / len(data)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:11:00.481261Z","iopub.execute_input":"2023-06-29T19:11:00.481688Z","iopub.status.idle":"2023-06-29T19:11:00.489881Z","shell.execute_reply.started":"2023-06-29T19:11:00.481655Z","shell.execute_reply":"2023-06-29T19:11:00.488759Z"}}
data['Self_Employed'].mode()[0]

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:38:51.642588Z","iopub.execute_input":"2023-06-29T19:38:51.643023Z","iopub.status.idle":"2023-06-29T19:38:51.649275Z","shell.execute_reply.started":"2023-06-29T19:38:51.642991Z","shell.execute_reply":"2023-06-29T19:38:51.648150Z"}}
data['Self_Employed']=data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:11:48.753997Z","iopub.execute_input":"2023-06-29T19:11:48.754377Z","iopub.status.idle":"2023-06-29T19:11:48.766680Z","shell.execute_reply.started":"2023-06-29T19:11:48.754350Z","shell.execute_reply":"2023-06-29T19:11:48.765358Z"}}
data.isnull().sum()*100 / len(data)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:12:05.621235Z","iopub.execute_input":"2023-06-29T19:12:05.621650Z","iopub.status.idle":"2023-06-29T19:12:05.629545Z","shell.execute_reply.started":"2023-06-29T19:12:05.621591Z","shell.execute_reply":"2023-06-29T19:12:05.628172Z"}}
data['Gender'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:12:18.524834Z","iopub.execute_input":"2023-06-29T19:12:18.525254Z","iopub.status.idle":"2023-06-29T19:12:18.532344Z","shell.execute_reply.started":"2023-06-29T19:12:18.525222Z","shell.execute_reply":"2023-06-29T19:12:18.531547Z"}}
data['Self_Employed'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:12:29.956533Z","iopub.execute_input":"2023-06-29T19:12:29.957423Z","iopub.status.idle":"2023-06-29T19:12:29.965646Z","shell.execute_reply.started":"2023-06-29T19:12:29.957387Z","shell.execute_reply":"2023-06-29T19:12:29.964480Z"}}
data['Credit_History'].mode()[0]

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:39:01.211690Z","iopub.execute_input":"2023-06-29T19:39:01.212109Z","iopub.status.idle":"2023-06-29T19:39:01.219274Z","shell.execute_reply.started":"2023-06-29T19:39:01.212078Z","shell.execute_reply":"2023-06-29T19:39:01.218072Z"}}
data['Credit_History']=data['Credit_History'].fillna(data['Credit_History'].mode()[0])

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:12:57.696364Z","iopub.execute_input":"2023-06-29T19:12:57.696795Z","iopub.status.idle":"2023-06-29T19:12:57.710217Z","shell.execute_reply.started":"2023-06-29T19:12:57.696755Z","shell.execute_reply":"2023-06-29T19:12:57.709091Z"}}
data.isnull().sum()*100 / len(data)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:17:06.535096Z","iopub.execute_input":"2023-06-29T19:17:06.536319Z","iopub.status.idle":"2023-06-29T19:17:06.557315Z","shell.execute_reply.started":"2023-06-29T19:17:06.536280Z","shell.execute_reply":"2023-06-29T19:17:06.556127Z"}}
data.sample(5)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:39:07.512758Z","iopub.execute_input":"2023-06-29T19:39:07.513174Z","iopub.status.idle":"2023-06-29T19:39:07.519395Z","shell.execute_reply.started":"2023-06-29T19:39:07.513145Z","shell.execute_reply":"2023-06-29T19:39:07.518440Z"}}
data['Dependents']=data['Dependents'].replace(to_replace="3+",value='4')

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:17:35.820646Z","iopub.execute_input":"2023-06-29T19:17:35.821744Z","iopub.status.idle":"2023-06-29T19:17:35.829960Z","shell.execute_reply.started":"2023-06-29T19:17:35.821706Z","shell.execute_reply":"2023-06-29T19:17:35.828643Z"}}
data['Dependents'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:17:52.978856Z","iopub.execute_input":"2023-06-29T19:17:52.979254Z","iopub.status.idle":"2023-06-29T19:17:52.987262Z","shell.execute_reply.started":"2023-06-29T19:17:52.979219Z","shell.execute_reply":"2023-06-29T19:17:52.985949Z"}}
data['Loan_Status'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2023-06-30T14:00:15.060006Z","iopub.execute_input":"2023-06-30T14:00:15.060331Z","iopub.status.idle":"2023-06-30T14:00:15.410406Z","shell.execute_reply.started":"2023-06-30T14:00:15.060307Z","shell.execute_reply":"2023-06-30T14:00:15.409392Z"}}
data.dropna(inplace=True)
data['Gender'] = data['Gender'].map({'Male':1,'Female':0}).astype('int')
data['Married'] = data['Married'].map({'Yes':1,'No':0}).astype('int')
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
data['Property_Area'] = data['Property_Area'].map({'Rural':0,'Semiurban':2,'Urban':1}).astype('int')
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0}).astype('int')

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:18:26.069491Z","iopub.execute_input":"2023-06-29T19:18:26.069926Z","iopub.status.idle":"2023-06-29T19:18:26.088292Z","shell.execute_reply.started":"2023-06-29T19:18:26.069889Z","shell.execute_reply":"2023-06-29T19:18:26.086995Z"}}


data.head()



# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:25:42.442465Z","iopub.execute_input":"2023-06-29T19:25:42.442849Z","iopub.status.idle":"2023-06-29T19:25:42.449236Z","shell.execute_reply.started":"2023-06-29T19:25:42.442818Z","shell.execute_reply":"2023-06-29T19:25:42.448052Z"}}
X = data.drop('Loan_Status',axis=1)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:18:57.476450Z","iopub.execute_input":"2023-06-29T19:18:57.476895Z","iopub.status.idle":"2023-06-29T19:18:57.482960Z","shell.execute_reply.started":"2023-06-29T19:18:57.476862Z","shell.execute_reply":"2023-06-29T19:18:57.481579Z"}}
y = data['Loan_Status']

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:19:26.791782Z","iopub.execute_input":"2023-06-29T19:19:26.792418Z","iopub.status.idle":"2023-06-29T19:19:26.801026Z","shell.execute_reply.started":"2023-06-29T19:19:26.792386Z","shell.execute_reply":"2023-06-29T19:19:26.799657Z"}}
y

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:19:58.942963Z","iopub.execute_input":"2023-06-29T19:19:58.943349Z","iopub.status.idle":"2023-06-29T19:19:58.961914Z","shell.execute_reply.started":"2023-06-29T19:19:58.943322Z","shell.execute_reply":"2023-06-29T19:19:58.961076Z"}}
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:20:33.148563Z","iopub.execute_input":"2023-06-29T19:20:33.149043Z","iopub.status.idle":"2023-06-29T19:20:33.154130Z","shell.execute_reply.started":"2023-06-29T19:20:33.149012Z","shell.execute_reply":"2023-06-29T19:20:33.152746Z"}}
cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:25:50.602212Z","iopub.execute_input":"2023-06-29T19:25:50.602641Z","iopub.status.idle":"2023-06-29T19:25:50.613850Z","shell.execute_reply.started":"2023-06-29T19:25:50.602595Z","shell.execute_reply":"2023-06-29T19:25:50.612988Z"}}
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X[cols]=st.fit_transform(X[cols])



# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:25:54.853063Z","iopub.execute_input":"2023-06-29T19:25:54.853680Z","iopub.status.idle":"2023-06-29T19:25:54.878085Z","shell.execute_reply.started":"2023-06-29T19:25:54.853640Z","shell.execute_reply":"2023-06-29T19:25:54.877075Z"}}
X

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:27:16.822841Z","iopub.execute_input":"2023-06-29T19:27:16.823229Z","iopub.status.idle":"2023-06-29T19:27:16.943125Z","shell.execute_reply.started":"2023-06-29T19:27:16.823201Z","shell.execute_reply":"2023-06-29T19:27:16.941986Z"}}
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:27:31.003147Z","iopub.execute_input":"2023-06-29T19:27:31.003534Z","iopub.status.idle":"2023-06-29T19:27:31.011122Z","shell.execute_reply.started":"2023-06-29T19:27:31.003506Z","shell.execute_reply":"2023-06-29T19:27:31.009838Z"}}
model_df={}
def model_val(model,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,
                                                   test_size=0.20,
                                                   random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"{model} accuracy is {accuracy_score(y_test,y_pred)}")
    
    score = cross_val_score(model,X,y,cv=5)
    print(f"{model} Avg cross val score is {np.mean(score)}")
    model_df[model]=round(np.mean(score)*100,2)
    

# %% [code]
model_df

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:28:37.179851Z","iopub.execute_input":"2023-06-29T19:28:37.180824Z","iopub.status.idle":"2023-06-29T19:28:37.384318Z","shell.execute_reply.started":"2023-06-29T19:28:37.180788Z","shell.execute_reply":"2023-06-29T19:28:37.383180Z"}}
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model_val(model,X,y)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:28:55.536954Z","iopub.execute_input":"2023-06-29T19:28:55.537350Z","iopub.status.idle":"2023-06-29T19:28:55.656529Z","shell.execute_reply.started":"2023-06-29T19:28:55.537321Z","shell.execute_reply":"2023-06-29T19:28:55.655383Z"}}
from sklearn import svm
model = svm.SVC()
model_val(model,X,y)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:29:09.008367Z","iopub.execute_input":"2023-06-29T19:29:09.008757Z","iopub.status.idle":"2023-06-29T19:29:09.238007Z","shell.execute_reply.started":"2023-06-29T19:29:09.008726Z","shell.execute_reply":"2023-06-29T19:29:09.236934Z"}}
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model_val(model,X,y)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:29:24.291067Z","iopub.execute_input":"2023-06-29T19:29:24.292290Z","iopub.status.idle":"2023-06-29T19:29:26.080347Z","shell.execute_reply.started":"2023-06-29T19:29:24.292250Z","shell.execute_reply":"2023-06-29T19:29:26.079103Z"}}
from sklearn.ensemble import RandomForestClassifier
model =RandomForestClassifier()
model_val(model,X,y)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:29:46.587877Z","iopub.execute_input":"2023-06-29T19:29:46.588278Z","iopub.status.idle":"2023-06-29T19:29:47.516041Z","shell.execute_reply.started":"2023-06-29T19:29:46.588248Z","shell.execute_reply":"2023-06-29T19:29:47.514855Z"}}
from sklearn.ensemble import GradientBoostingClassifier
model =GradientBoostingClassifier()
model_val(model,X,y)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:30:10.217195Z","iopub.execute_input":"2023-06-29T19:30:10.217800Z","iopub.status.idle":"2023-06-29T19:30:10.221989Z","shell.execute_reply.started":"2023-06-29T19:30:10.217770Z","shell.execute_reply":"2023-06-29T19:30:10.220905Z"}}
from sklearn.model_selection import RandomizedSearchCV

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:30:23.590289Z","iopub.execute_input":"2023-06-29T19:30:23.590704Z","iopub.status.idle":"2023-06-29T19:30:23.596039Z","shell.execute_reply.started":"2023-06-29T19:30:23.590674Z","shell.execute_reply":"2023-06-29T19:30:23.594898Z"}}
log_reg_grid={"C":np.logspace(-4,4,20),
             "solver":['liblinear']}

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:30:40.378957Z","iopub.execute_input":"2023-06-29T19:30:40.379348Z","iopub.status.idle":"2023-06-29T19:30:40.385023Z","shell.execute_reply.started":"2023-06-29T19:30:40.379320Z","shell.execute_reply":"2023-06-29T19:30:40.383781Z"}}
rs_log_reg=RandomizedSearchCV(LogisticRegression(),
                   param_distributions=log_reg_grid,
                  n_iter=20,cv=5,verbose=True)

# %% [code]
rs_log_reg.fit(X,y)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:31:22.538200Z","iopub.execute_input":"2023-06-29T19:31:22.538587Z","iopub.status.idle":"2023-06-29T19:31:22.546065Z","shell.execute_reply.started":"2023-06-29T19:31:22.538558Z","shell.execute_reply":"2023-06-29T19:31:22.544880Z"}}
rs_log_reg.best_score_

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:31:37.468364Z","iopub.execute_input":"2023-06-29T19:31:37.468793Z","iopub.status.idle":"2023-06-29T19:31:37.477946Z","shell.execute_reply.started":"2023-06-29T19:31:37.468763Z","shell.execute_reply":"2023-06-29T19:31:37.476561Z"}}
rs_log_reg.best_params_

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:31:52.452004Z","iopub.execute_input":"2023-06-29T19:31:52.452382Z","iopub.status.idle":"2023-06-29T19:31:52.458221Z","shell.execute_reply.started":"2023-06-29T19:31:52.452355Z","shell.execute_reply":"2023-06-29T19:31:52.457074Z"}}
svc_grid = {'C':[0.25,0.50,0.75,1],"kernel":["linear"]}

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:32:10.133774Z","iopub.execute_input":"2023-06-29T19:32:10.134198Z","iopub.status.idle":"2023-06-29T19:32:10.139890Z","shell.execute_reply.started":"2023-06-29T19:32:10.134163Z","shell.execute_reply":"2023-06-29T19:32:10.138564Z"}}
rs_svc=RandomizedSearchCV(svm.SVC(),
                  param_distributions=svc_grid,
                   cv=5,
                   n_iter=20,
                  verbose=True)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:32:29.087096Z","iopub.execute_input":"2023-06-29T19:32:29.087505Z","iopub.status.idle":"2023-06-29T19:32:29.349465Z","shell.execute_reply.started":"2023-06-29T19:32:29.087474Z","shell.execute_reply":"2023-06-29T19:32:29.348285Z"}}
rs_svc.fit(X,y)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:32:43.002796Z","iopub.execute_input":"2023-06-29T19:32:43.003206Z","iopub.status.idle":"2023-06-29T19:32:43.011023Z","shell.execute_reply.started":"2023-06-29T19:32:43.003175Z","shell.execute_reply":"2023-06-29T19:32:43.009864Z"}}
rs_svc.best_score_

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:32:57.760480Z","iopub.execute_input":"2023-06-29T19:32:57.760925Z","iopub.status.idle":"2023-06-29T19:32:57.769395Z","shell.execute_reply.started":"2023-06-29T19:32:57.760895Z","shell.execute_reply":"2023-06-29T19:32:57.768104Z"}}
rs_svc.best_params_

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:33:14.659000Z","iopub.execute_input":"2023-06-29T19:33:14.659380Z","iopub.status.idle":"2023-06-29T19:33:14.668779Z","shell.execute_reply.started":"2023-06-29T19:33:14.659351Z","shell.execute_reply":"2023-06-29T19:33:14.667524Z"}}
RandomForestClassifier()

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:33:29.964223Z","iopub.execute_input":"2023-06-29T19:33:29.964711Z","iopub.status.idle":"2023-06-29T19:33:29.971804Z","shell.execute_reply.started":"2023-06-29T19:33:29.964677Z","shell.execute_reply":"2023-06-29T19:33:29.970550Z"}}
rf_grid={'n_estimators':np.arange(10,1000,10),
  'max_features':['auto','sqrt'],
 'max_depth':[None,3,5,10,20,30],
 'min_samples_split':[2,5,20,50,100],
 'min_samples_leaf':[1,2,5,10]
 }

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:33:43.185319Z","iopub.execute_input":"2023-06-29T19:33:43.185772Z","iopub.status.idle":"2023-06-29T19:33:43.191327Z","shell.execute_reply.started":"2023-06-29T19:33:43.185737Z","shell.execute_reply":"2023-06-29T19:33:43.190123Z"}}
rs_rf=RandomizedSearchCV(RandomForestClassifier(),
                  param_distributions=rf_grid,
                   cv=5,
                   n_iter=20,
                  verbose=True)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:56:51.259435Z","iopub.execute_input":"2023-06-29T19:56:51.259895Z","iopub.status.idle":"2023-06-29T19:56:56.181546Z","shell.execute_reply.started":"2023-06-29T19:56:51.259862Z","shell.execute_reply":"2023-06-29T19:56:56.179666Z"}}
rs_rf.fit(X,y)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:37:40.944002Z","iopub.execute_input":"2023-06-29T19:37:40.944406Z","iopub.status.idle":"2023-06-29T19:37:40.980181Z","shell.execute_reply.started":"2023-06-29T19:37:40.944376Z","shell.execute_reply":"2023-06-29T19:37:40.978340Z"}}
rs_rf.best_score_

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:57:17.649653Z","iopub.execute_input":"2023-06-29T19:57:17.650070Z","iopub.status.idle":"2023-06-29T19:57:17.689303Z","shell.execute_reply.started":"2023-06-29T19:57:17.650037Z","shell.execute_reply":"2023-06-29T19:57:17.688258Z"}}
rs_rf.best_params_

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:57:53.778909Z","iopub.execute_input":"2023-06-29T19:57:53.779288Z","iopub.status.idle":"2023-06-29T19:57:53.789377Z","shell.execute_reply.started":"2023-06-29T19:57:53.779260Z","shell.execute_reply":"2023-06-29T19:57:53.787767Z"}}
LogisticRegression score Before Hyperparameter Tuning: 80.48
LogisticRegression score after Hyperparameter Tuning: 80.48 
    
------------------------------------------------------
SVC score Before Hyperparameter Tuning: 79.38
SVC score after Hyperparameter Tuning: 80.66
    
--------------------------------------------------------
RandomForestClassifier score Before Hyperparameter Tuning: 77.76
RandomForestClassifier score after Hyperparameter Tuning: 80.66 

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:58:37.529263Z","iopub.execute_input":"2023-06-29T19:58:37.529712Z","iopub.status.idle":"2023-06-29T19:58:37.536002Z","shell.execute_reply.started":"2023-06-29T19:58:37.529679Z","shell.execute_reply":"2023-06-29T19:58:37.534735Z"}}
X = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:58:49.585927Z","iopub.execute_input":"2023-06-29T19:58:49.586348Z","iopub.status.idle":"2023-06-29T19:58:49.591990Z","shell.execute_reply.started":"2023-06-29T19:58:49.586317Z","shell.execute_reply":"2023-06-29T19:58:49.590600Z"}}
rf = RandomForestClassifier(n_estimators=270,
 min_samples_split=5,
 min_samples_leaf=5,
 max_features='sqrt',
 max_depth=5)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:59:05.185541Z","iopub.execute_input":"2023-06-29T19:59:05.185993Z","iopub.status.idle":"2023-06-29T19:59:05.853864Z","shell.execute_reply.started":"2023-06-29T19:59:05.185961Z","shell.execute_reply":"2023-06-29T19:59:05.852430Z"}}


rf.fit(X,y)



# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:59:20.190622Z","iopub.execute_input":"2023-06-29T19:59:20.191060Z","iopub.status.idle":"2023-06-29T19:59:20.196161Z","shell.execute_reply.started":"2023-06-29T19:59:20.191020Z","shell.execute_reply":"2023-06-29T19:59:20.194962Z"}}


import joblib



# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:59:31.034100Z","iopub.execute_input":"2023-06-29T19:59:31.034992Z","iopub.status.idle":"2023-06-29T19:59:31.242304Z","shell.execute_reply.started":"2023-06-29T19:59:31.034959Z","shell.execute_reply":"2023-06-29T19:59:31.241328Z"}}
joblib.dump(rf,'loan_status_predict')

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T19:59:46.429872Z","iopub.execute_input":"2023-06-29T19:59:46.430297Z","iopub.status.idle":"2023-06-29T19:59:46.572402Z","shell.execute_reply.started":"2023-06-29T19:59:46.430264Z","shell.execute_reply":"2023-06-29T19:59:46.571136Z"}}
model = joblib.load('loan_status_predict')

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T20:00:01.894359Z","iopub.execute_input":"2023-06-29T20:00:01.894826Z","iopub.status.idle":"2023-06-29T20:00:01.903714Z","shell.execute_reply.started":"2023-06-29T20:00:01.894791Z","shell.execute_reply":"2023-06-29T20:00:01.902410Z"}}
import pandas as pd
df = pd.DataFrame({
    'Gender':1,
    'Married':1,
    'Dependents':2,
    'Education':0,
    'Self_Employed':0,
    'ApplicantIncome':2889,
    'CoapplicantIncome':0.0,
    'LoanAmount':45,
    'Loan_Amount_Term':180,
    'Credit_History':0,
    'Property_Area':1
},index=[0])

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T20:00:13.643467Z","iopub.execute_input":"2023-06-29T20:00:13.643908Z","iopub.status.idle":"2023-06-29T20:00:13.656780Z","shell.execute_reply.started":"2023-06-29T20:00:13.643877Z","shell.execute_reply":"2023-06-29T20:00:13.656029Z"}}
df

# %% [code]
result = model.predict(df)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T20:00:39.358276Z","iopub.execute_input":"2023-06-29T20:00:39.359470Z","iopub.status.idle":"2023-06-29T20:00:39.399914Z","shell.execute_reply.started":"2023-06-29T20:00:39.359418Z","shell.execute_reply":"2023-06-29T20:00:39.398142Z"}}
if result==1:
    print("Loan Approved")
else:
    print("Loan Not Approved")

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T20:00:58.193160Z","iopub.execute_input":"2023-06-29T20:00:58.193583Z","iopub.status.idle":"2023-06-29T20:00:58.232289Z","shell.execute_reply.started":"2023-06-29T20:00:58.193547Z","shell.execute_reply":"2023-06-29T20:00:58.231148Z"}}
from tkinter import *
import joblib
import pandas as pd

# %% [code] {"execution":{"iopub.status.busy":"2023-06-29T20:01:22.644640Z","iopub.execute_input":"2023-06-29T20:01:22.645046Z","iopub.status.idle":"2023-06-29T20:01:23.251511Z","shell.execute_reply.started":"2023-06-29T20:01:22.645011Z","shell.execute_reply":"2023-06-29T20:01:23.249959Z"}}
def show_entry():
    
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    p7 = float(e7.get())
    p8 = float(e8.get())
    p9 = float(e9.get())
    p10 = float(e10.get())
    p11 = float(e11.get())
    
    model = joblib.load('loan_status_predict')
    df = pd.DataFrame({
    'Gender':p1,
    'Married':p2,
    'Dependents':p3,
    'Education':p4,
    'Self_Employed':p5,
    'ApplicantIncome':p6,
    'CoapplicantIncome':p7,
    'LoanAmount':p8,
    'Loan_Amount_Term':p9,
    'Credit_History':p10,
    'Property_Area':p11
},index=[0])
    result = model.predict(df)
    
    if result == 1:
        Label(master, text="Loan approved").grid(row=31)
    else:
        Label(master, text="Loan Not Approved").grid(row=31)
        
    
master =Tk()
master.title("Loan Status Prediction Using Machine Learning")
label = Label(master,text = "Loan Status Prediction",bg = "black",
               fg = "white").grid(row=0,columnspan=2)

Label(master,text = "Gender [1:Male ,0:Female]").grid(row=1)
Label(master,text = "Married [1:Yes,0:No]").grid(row=2)
Label(master,text = "Dependents [1,2,3,4]").grid(row=3)
Label(master,text = "Education").grid(row=4)
Label(master,text = "Self_Employed").grid(row=5)
Label(master,text = "ApplicantIncome").grid(row=6)
Label(master,text = "CoapplicantIncome").grid(row=7)
Label(master,text = "LoanAmount").grid(row=8)
Label(master,text = "Loan_Amount_Term").grid(row=9)
Label(master,text = "Credit_History").grid(row=10)
Label(master,text = "Property_Area").grid(row=11)


e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)


e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)
e7.grid(row=7,column=1)
e8.grid(row=8,column=1)
e9.grid(row=9,column=1)
e10.grid(row=10,column=1)
e11.grid(row=11,column=1)

Button(master,text="Predict",command=show_entry).grid()

mainloop()

# %% [code]
