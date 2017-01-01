import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
import xgboost

loc_train = "train.csv"
loc_test = "test.csv"
loc_submission = "kaggle.rf200.entropy.submission.csv"

df_train = pd.read_csv(loc_train)
df_test = pd.read_csv(loc_test)

feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

X_train = df_train[feature_cols]
X_test = df_test[feature_cols]
y = df_train['Cover_Type']
test_ids = df_test['Id']
del df_train
del df_test





clf = ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X_train,y ,test_size=0.1, random_state=0)
clf.fit(X_train, y_train)

# fit model no training data
xgboost = xgboost.XGBClassifier()
xgboost.fit(X_train, y_train)

print clf.score(X_test, y_test)  
print xgboost.score(X_test, y_test)  
del X_train

# with open(loc_submission, "w") as outfile:
#   outfile.write("Id,Cover_Type\n")
#   for e, val in enumerate(list(clf.predict(X_test))):
#     outfile.write("%s,%s\n"%(test_ids[e],val))