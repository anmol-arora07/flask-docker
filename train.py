import json
import os
from importlib_metadata import metadata

from joblib import dump

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import numpy as np

MODEL_DIR=os.environ["MODEL_DIR"]
MODEL_FILE=os.environ["MODEL_FILE"]
METADATA_FILE=os.environ["METADATA_FILE"]
MODEL_PATH=os.path.join(MODEL_DIR,MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR,METADATA_FILE)

## LOAD AND SPLIT DATA ##

print("Loading Data ..")

boston=datasets.load_boston()
X,y=shuffle(boston.data,boston.target,random_state=13)

X=X.astype(np.float32)
offset=int(X.shape[0]*0.9)
X_train, y_train=X[:offset],y[:offset]
X_test,y_test=X[offset:],y[offset:] 

## FIT REGRESSION MODEL ##
print("Fitting model...")
params= {'n_estimators':500,'max_depth':4,'min_samples_split':2,'learning_rate': 0.01,'loss':'ls'}

clf=ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train,y_train)

train_mse= mean_squared_error(y_train,clf.predict(X_train))
test_mse= mean_squared_error(y_test,clf.predict(X_test))

metadata= {
    "train_mean_squared_error": train_mse,
    "test_mean_squared_error": test_mse
}


## SERIALIZE MODELA AND METADATA ##
print(f"Serializing model to : {MODEL_PATH}")
dump(clf,MODEL_PATH)

print(f"Serializing metadata to: {METADATA_PATH}")
with open(METADATA_PATH,'w') as outfile:
    json.dump(metadata,outfile)


