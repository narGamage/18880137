

import numpy as np
import pandas as pd

from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns




    
df = pd.read_csv("SL_contributing.csv")
print(df.head())

df.fillna(999, inplace=True)
X=df.drop(labels=['Cardio'],axis='columns')
y=df['Cardio']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
sss=X_train


from sklearn.decomposition import PCA

pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
sss1=X_train


explained_variance = pca.explained_variance_ratio_

Find_Most_Sutable=abs( pca.components_ )


from sklearn.decomposition import PCA
 
pca = PCA(n_components=15)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes  import GaussianNB


models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('GradientBoostingClassifier', GradientBoostingClassifier()))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
models.append(('Naive Bayes', GaussianNB()))


results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
    modell=model
    modell.fit(X_train,y_train)
    pred_y=modell.predict(X_test)

    cm = metrics.confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = str(model)+' Confusion Matrix - score:'+str(metrics.accuracy_score(y_test,pred_y))
    plt.title(all_sample_title, size = 15);
    plt.show()
    print(metrics.classification_report(y_test,pred_y))






