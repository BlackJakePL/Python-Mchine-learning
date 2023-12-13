import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import neurolab as neuro
import numpy.random as rand
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 

def MissingDataCheck():
    print("Podaj nazwe lub sciezke do pliku: ")
    file = input()
    df = pd.read_csv(file+".csv")
    print(df.head())
    print(df.info())

    counter=0
    for row in df.Glucose:
        if(row== 0):
            counter+=1
    idxs = df.loc[df['Glucose'] == 0] .index
    print("Brakujacych danych w kolumnie Glucose: ",counter)
    for i in idxs:
        tmp_blood_pressure = df.loc[[i]]['BloodPressure'].values[0]
        tmp = df.loc[(df['BloodPressure'] >= tmp_blood_pressure-10) & (df['BloodPressure'] <= tmp_blood_pressure+10)]
        mean_glucose = round(tmp['Glucose'].mean())
        print(mean_glucose)
        print(df.loc[i,'Glucose'])
        df.loc[i,'Glucose'] = mean_glucose
        print(df.loc[i,'Glucose'])
    print("uzupełniono brakujace dane w kolumnie Glucose")

    counter=0
    for row in df.BloodPressure:
        if(row== 0):
            counter+=1
    idxs = df.loc[df['BloodPressure'] == 0] .index
    print("Brakujacych danych w kolumnie BloodPressure: ",counter)
    for i in idxs:
        tmp_Age = df.loc[[i]]['Age'].values[0]
        tmp = df.loc[(df['BloodPressure'] >= tmp_Age-3) & (df['BloodPressure'] <= tmp_Age+3)]
        mean_BloodPressure = round(tmp['BloodPressure'].mean())
        print(mean_BloodPressure)
        print(df.loc[i,'BloodPressure'])
        df.loc[i,'BloodPressure'] = mean_BloodPressure
        print(df.loc[i,'BloodPressure'])
    print("uzupełniono brakujace dane w kolumnie BloodPressure")

    counter=0
    for row in df.SkinThickness:
        if(row== 0):
            counter+=1
    idxs = df.loc[df['SkinThickness'] == 0] .index
    print("Brakujacych danych w kolumnie SkinThickness: ",counter)
    for i in idxs:
        tmp_BMI = df.loc[[i]]['BMI'].values[0]
        tmp = df.loc[(df['BMI'] >= tmp_BMI-1) & (df['BMI'] <= tmp_BMI+1)]
        mean_SkinThickness = round(tmp['SkinThickness'].mean())
        print(mean_SkinThickness)
        print(df.loc[i,'SkinThickness'])
        df.loc[i,'SkinThickness'] = mean_SkinThickness
        print(df.loc[i,'SkinThickness'])
    print("uzupełniono brakujace dane w kolumnie SkinThickness")

    counter=0
    for row in df.BMI:
        if(row== 0):
            counter+=1
    idxs = df.loc[df['BMI'] == 0] .index
    print("Brakujacych danych w kolumnie BMI: ",counter)
    for i in idxs:
        tmp_BMI = df.loc[[i]]['SkinThickness'].values[0]
        tmp = df.loc[(df['SkinThickness'] >= tmp_BMI-5) & (df['SkinThickness'] <= tmp_BMI+5)]
        mean_SkinThickness = round(tmp['BMI'].mean())
        print(mean_SkinThickness)
        print(df.loc[i,'BMI'])
        df.loc[i,'BMI'] = mean_SkinThickness
        print(df.loc[i,'BMI'])
    print("uzupełniono brakujace dane w kolumnie BMI")

    counter=0
    for row in df.Insulin:
        if(row== 0):
            counter+=1
    idxs = df.loc[df['Insulin'] == 0] .index
    print("Brakujacych danych w kolumnie Insulin: ",counter)
    for i in idxs:
        tmp_Insulin = df.loc[[i]]['Glucose'].values[0]
        tmp = df.loc[(df['Glucose'] >= tmp_Insulin-13) & (df['Glucose'] <= tmp_Insulin+13)]
        mean_Insulin = round(tmp['Insulin'].mean())
        print(mean_Insulin)
        print(df.loc[i,'Insulin'])
        df.loc[i,'Insulin'] = mean_Insulin
        print(df.loc[i,'Insulin'])
    print("uzupełniono brakujace dane ")
    print("podaj nazwe nowego pliku: ")
    name= input()
    df.to_csv(name+'.csv')

def NormalizeDataSet():
    print("Podaj nazwe lub sciezke do pliku: ")
    file = input()
    df = pd.read_csv(file+".csv")
    print(df.head())
    print(df.info())

    df['Pregnancies'] /= np.linalg.norm(df['Pregnancies'])
    df['Glucose'] /= np.linalg.norm(df['Glucose'])
    df['BloodPressure'] /= np.linalg.norm(df['BloodPressure'])
    df['SkinThickness'] /= np.linalg.norm(df['SkinThickness'])
    df['Insulin'] /= np.linalg.norm(df['Insulin'])
    df['BMI'] /= np.linalg.norm(df['BMI'])
    df['DiabetesPedigreeFunction'] /= np.linalg.norm(df['DiabetesPedigreeFunction'])
    df['Age'] /= np.linalg.norm(df['Age'])
    print("Znormalizowane dane w pliku csv")
    print("podaj nazwe nowego pliku: ")
    name= input()
    df.to_csv(name+'.csv')

def LearnandtestFeedForward():
    print("Podaj nazwe lub sciezke do pliku: ")
    file = input()
    df = pd.read_csv(file+".csv")
    df = df.drop(df.columns[[0,1]], axis=1)
    print(df.head())
    print(df.info())
    vals = df.values
    print(vals)
    learn=vals[0:500, 0:-1]
    learn_out= vals[0:500, -1].reshape(500,1)
    print(learn.shape)
    test = vals[500:,0:-1]
    test_out = vals[500: ,-1]
    print(test_out.shape)

    neural = neuro.net.newff([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],[30,10,1])
    err = neural.train(learn, learn_out, show=1)

    simulation = neural.sim(test)
    simulation[simulation<0.5]=0
    simulation[simulation>=0.5]=1
    simulation = simulation.flatten()
    print(simulation[:10])
    print(test_out[:10])

    cos_sim = (dot(simulation, test_out)/(norm(simulation)*norm(test_out))) * 100

    print(cos_sim ,"%")

def LearnandtestKNN():
    print("Podaj nazwe lub sciezke do pliku: ")
    file = input()
    df = pd.read_csv(file+".csv")
    df = df.drop(df.columns[[0, 1]], axis=1)
    print(df.head())
    print(df.info())

    X = df.drop('Outcome', axis=1) 
    y = df['Outcome']
    print('X: ', str(X.shape)) 
    print('Y: ', str(y.shape)) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state=42, shuffle=True)

    print('Wymiary X_train :', str(X_train.shape))
    print('Wymiary X_test est:', str(X_test.shape))
    print('Wymiary y_train est:', str(y_train.shape))
    print('Wymiary y_test est:', str(y_test.shape))

    knn      = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',metric='minkowski')
    knn_fit  = knn.fit(X_train, y_train)

    knn_pred = knn.predict(X_test)

    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)
    print('Dokładność klasyfikacji modelu KNN (Training set):', str(knn.score(X_train,y_train)))
    print('Dokładność klasyfikacji modelu KNN (Test set):', str(knn.score(X_test,y_test)))

    K_numbers = np.arange(1,26)

    train_acc = []
    test_acc  = [] 
    
    for  k in K_numbers:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        knn.predict(X_test)
        train_acc.append(knn.score(X_train, y_train))
        test_acc.append(knn.score(X_test, y_test))

    plt.figure(figsize=(10,5),edgecolor='black', facecolor='deepskyblue')
    plt.plot(K_numbers, train_acc, color = 'Blue', lw=3, marker='+', label='Dokładność treningu')
    plt.plot(K_numbers, test_acc, color = 'Black', lw=3, marker='+', linestyle='--' ,label='Test dokladnosci')
    plt.title('Wybór K w oparciu o dokładność modelu (KNN)', fontsize=14, fontweight='bold')
    plt.xlabel('Liczba K', fontsize=12)
    plt.ylabel('Poziom dokładności', fontsize=12)
    plt.legend(loc = 0)
    plt.grid(True, lw=0.3)
    plt.axhline(y=max(test_acc), color='red', alpha=0.5)


    plt.show()

    knn_CV = KNeighborsClassifier(n_neighbors=5)
    KF = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(knn_CV, X, y, cv=KF, scoring='accuracy')

    print('Średnia precyzja bez walidacji krzyżowej dla K = 5:', "{:.0%}".format(test_accuracy))
    print('Średnia dokładność w ciągu 5-krotności dla K = 5:', "{:.0%}".format(np.mean(cv_scores)))
    print('Zwiększono dokładność modelu o:',"{:.0%}".format((np.mean(cv_scores)-test_accuracy)))

    K_numbers = np.arange(1,26)

    test_acc_CV = []

    for i in K_numbers:
        KF_CV = KFold(n_splits=5, random_state=42, shuffle=True)
        knn = KNeighborsClassifier(n_neighbors=i)
        knn_cv_scores = cross_val_score(knn, X, y, cv=KF_CV, scoring='accuracy')
        test_acc_CV.append(np.mean(knn_cv_scores))


    plt.figure(figsize=(10,5),edgecolor='black', facecolor='deepskyblue')
    plt.plot(K_numbers, test_acc_CV, color = 'Black', lw=3, marker='o', linestyle='--' ,label='Test dokladnosci')
    plt.plot(K_numbers, test_acc, color = 'Blue', lw=1.5, marker='+', linestyle='--' ,label='Test dokladnosci bez walidacji krzyzowej')
    plt.fill_between(K_numbers, test_acc_CV, test_acc, color='red', alpha=0.2)
    plt.title('Najlepszy wybór K w oparciu o dokładność modelu (KNN)', fontsize=14, fontweight='bold')
    plt.xlabel('Wartosc K', fontsize=12)
    plt.ylabel('Poziom dokładności', fontsize=12)
    plt.legend(loc = 0)
    plt.grid(True, lw=0.3) 

    plt.show()

    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 26)}
    knn_gscv = GridSearchCV(knn, param_grid, cv=5)
    knn_gscv.fit(X, y)
    knn_gscv.best_params_
    knn_gscv.best_score_

    print('Najoptymalniejsza wartosc K: ', knn_gscv.best_params_)

    plt.figure(figsize=(6,6))
    y1 = df['Outcome'].values
    X1 = df.drop('Outcome', axis=1).values
    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X1, y1, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train_n)
    X_test_sc = sc.transform(X_test_n)
    clf_knn = KNeighborsClassifier(n_neighbors=7) 
    pca = PCA(n_components = 2)
    X_train_pca = pca.fit_transform(X_train_sc)
    clf_knn.fit(X_train_pca, y_train_n)
    plot_decision_regions(X_train_pca, y_train_n, clf=clf_knn, legend=2)
    print("Średnia skutecznosc podanego zbioru",clf_knn.score(X_train_pca, y_train_n))
    plt.show()

while(True): 
    print("Wybierz opcje: ")
    print("1. Sprawdzenie brakujacych danych i uzupełnienie ich ")
    print("2. Normalizacja zbioru danych")
    print("3. Algorytm FeedForward")
    print("4. Algorytm KNN")
    x = input()
    if(x=="1"):
        MissingDataCheck()
    if(x=="2"):
        NormalizeDataSet()
    if(x=="3"):
        LearnandtestFeedForward()
    if(x=="4"):
        LearnandtestKNN()