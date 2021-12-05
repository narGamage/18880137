# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:42:14 2021

@author: dushm
"""

from tkinter import *
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
window = Tk()

def clicked():
    #Enter Details
    Eage=int(txtAge.get())
    Egender=Gender.get();
    Eheight=float(txtHeight.get())
    Eweight=float(txtWeight.get())
    Eap_hi=float(txtSBP.get())
    Eap_li=float(txtDBP.get())
    Echolesterol=Cholesterol.get()
    Egluc=Glucose.get()
    Esmoke=Smoking.get()
    Ealco=Alcohol.get()
    Eactive=Physical.get()
    
    print([[Eage,Egender,Eheight,Eweight,Eap_hi,Eap_li,Echolesterol,Egluc,Esmoke,Ealco,Eactive]])


    dataset = pd.read_csv("PCA_v1.csv")
    #dataset = pd.read_csv(url, names=names)
    dataset.head()
    X = dataset.drop(labels=['cardio'],axis=1)
    y = dataset['cardio']


    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    GivenData= sc.transform([[Eage,Egender,Eheight,Eweight,Eap_hi,Eap_li,Echolesterol,Egluc,Esmoke,Ealco,Eactive]])


    from sklearn.decomposition import PCA

    pca = PCA()
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    GivenData= pca.transform(GivenData)

    explained_variance = pca.explained_variance_ratio_

    from sklearn.decomposition import PCA

    pca = PCA(n_components=1)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    GivenData= pca.transform(GivenData)

    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier



    models = []
    models.append(('RandomForestClassifier', RandomForestClassifier()))



    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=0)
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
        print('---------------------Given Value Predict Result-----------------')
        print(modell.predict(GivenData))
        
        if (str(modell.predict(GivenData))=="[2]"):
            res="possibility of cvd occurrences : - Yes"
        else:
            res="possibility of cvd occurrences : -No"
            
        
    

    lblResult.configure(text= res)

window.title("CVD Predication")

lbl = Label(window, text="CVD Predication", font=("Arial Bold", 36))

lbl.grid(column=1, row=0)

lbl = Label(window, text="Enter Your Details :", font=("Arial Bold", 18))

lbl.grid(column=1, row=3)

lbl = Label(window, text="Age :                                                     ", font=("Arial Bold", 12),anchor='nw')

lbl.grid(column=0, row=4)

txtAge = Entry(window,width=25)

txtAge.grid(column=1, row=4)


lbl = Label(window, text="Gender   :                                              ", font=("Arial Bold", 12))

lbl.grid(column=0, row=5)

Gender = IntVar()

radGender1 = Radiobutton(window,text='M', value=1, variable=Gender)

radGender2 = Radiobutton(window,text='F', value=2, variable=Gender)

radGender1.grid(column=1, row=5)

radGender2.grid(column=1, row=6)



lbl = Label(window, text="Height (cm) :                                         ", font=("Arial Bold", 12))

lbl.grid(column=0, row=7)

txtHeight = Entry(window,width=25)

txtHeight.grid(column=1, row=7)

lbl = Label(window, text="Weight (kg):                                          ", font=("Arial Bold", 12))

lbl.grid(column=0, row=8)

txtWeight = Entry(window,width=25)

txtWeight.grid(column=1, row=8)

lbl = Label(window, text="Systolic blood pressure (mmHg): ", font=("Arial Bold", 12))

lbl.grid(column=0, row=9)

txtSBP = Entry(window,width=25)

txtSBP.grid(column=1, row=9)

lbl = Label(window, text="Diastolic blood pressure (mmHg):", font=("Arial Bold", 12))

lbl.grid(column=0, row=10)

txtDBP = Entry(window,width=25)

txtDBP.grid(column=1, row=10)

lbl = Label(window, text="Cholesterol  :                                        ", font=("Arial Bold", 12))

lbl.grid(column=0, row=11)

Cholesterol = IntVar()

radCholesterol1 = Radiobutton(window,text=' 1: normal', value=1, variable=Cholesterol)

radCholesterol2 = Radiobutton(window,text='2: above normal', value=2, variable=Cholesterol)

radCholesterol3 = Radiobutton(window,text='3: well above normal', value=3, variable=Cholesterol)

radCholesterol1.grid(column=1, row=11)

radCholesterol2.grid(column=1, row=12)

radCholesterol3.grid(column=1, row=13)


lbl = Label(window, text="Glucose   :                                              ", font=("Arial Bold", 12))

lbl.grid(column=0, row=14)

Glucose = IntVar()

radGlucose1 = Radiobutton(window,text=' 1: normal', value=1, variable=Glucose)

radGlucose2 = Radiobutton(window,text='2: above normal', value=2, variable=Glucose)

radGlucose3 = Radiobutton(window,text='3: well above normal', value=3, variable=Glucose)

radGlucose1.grid(column=1, row=14)

radGlucose2.grid(column=1, row=15)

radGlucose3.grid(column=1, row=16)

lbl = Label(window, text="Smoking    :                                           ", font=("Arial Bold", 12))

lbl.grid(column=0, row=17)

Smoking = IntVar()

radSmoking1 = Radiobutton(window,text='Yes', value=1, variable=Smoking)

radSmoking2 = Radiobutton(window,text='No', value=2, variable=Smoking)

radSmoking1.grid(column=1, row=17)

radSmoking2.grid(column=1, row=18)

lbl = Label(window, text="Alcohol intake    :                                  ", font=("Arial Bold", 12))

lbl.grid(column=0, row=19)

Alcohol = IntVar()

radAlcohol1 = Radiobutton(window,text='Yes', value=1, variable=Alcohol)

radAlcohol2 = Radiobutton(window,text='No', value=2, variable=Alcohol)

radAlcohol1.grid(column=1, row=19)

radAlcohol2.grid(column=1, row=20)


lbl = Label(window, text="Physical activity   :                             ", font=("Arial Bold", 12))

lbl.grid(column=0, row=21)

Physical = IntVar()

radPhysical1 = Radiobutton(window,text='Yes', value=1, variable=Physical)

radPhysical2 = Radiobutton(window,text='No', value=2, variable=Physical)

radPhysical1.grid(column=1, row=21)

radPhysical2.grid(column=1, row=22)


lbl = Label(window, text="-----------------------------------------------", font=("Arial Bold", 12))

lbl.grid(column=0, row=29)

btn = Button(window, text="Click Me", command=clicked)

btn.grid(column=1, row=29)

lblResult = Label(window, text="Result :- ", font=("Arial Bold", 12))

lblResult.grid(column=0, row=31)

window.mainloop()