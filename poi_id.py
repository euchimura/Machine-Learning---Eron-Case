#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import pickle
sys.path.append("../tools/")

import p5aux #Biblioteca auxiliar criado por Eric.
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import testador # Baseado no arquivo tester.py fornecido pela udacity

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV

import prettytable
from prettytable import ALL as ALL

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

## incicia as features list retirando as features com NAN > 50%
features_list = p5aux.inicia_features()

features_list_limpo = p5aux.inicia_features_limpo()

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

#Chama função que está na biblioteca criada: p5aux.py
#remove os outliers Total e 'the traval agency in the park'
p5aux.remove_outliers(data_dict)

# cria os atributos 'por_from_poi' e 'por_to_poi'
p5aux.cria_atributos(data_dict)

my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### cria uma tabela de resultados
tabela_resultado = p5aux.inicia_resultado()
tabela_teste = p5aux.inicia_tabela_testes()

## divide a amostragem entre treino e tester
features_train, features_test, labels_train, labels_test = train_test_split(
                    features,
                     labels,
                     test_size=0.3,
                     random_state=60)
cv = StratifiedShuffleSplit(labels_train,
                                   100,
                                   test_size=0.2,
                                   random_state=60)

## para cada classificador e testador serão gerados uma procedure

## Logistic regression
def classificador_logistic_regression():

    kb = SelectKBest()
    pca = PCA()
    clf_lr = LogisticRegression()

    pipe_lr = Pipeline(steps=[("SKB", kb), ("PCA", pca),
                              ("LogisticRegression", clf_lr)])

    lr_k = {"SKB__k": range(9, 10)}
    lr_params = {'LogisticRegression__C': [1e-08, 1e-07, 1e-06],
                 'LogisticRegression__tol': [1e-2, 1e-3, 1e-4],
                 'LogisticRegression__penalty': ['l1', 'l2'],
                 'LogisticRegression__random_state': [42, 46, 60]}
    lr_pca = {"PCA__n_components": range(3, 8), "PCA__whiten": [True, False]}

    lr_k.update(lr_params)
    lr_k.update(lr_pca)
    ## inicia o gridsearch
    grid_search = GridSearchCV(pipe_lr,
                               lr_k,
                               n_jobs=-1,
                               cv=cv,
                               scoring='f1')
    grid_search.fit(features_train, labels_train)
    #imprime resultados em uma tabela
    p5aux.imprime_resultado(tabela_resultado,
                            grid_search,
                            lr_k,
                            "Logistic Regression")

def classificador_svc():
    # inicia as variaveis
    kb = SelectKBest()
    pca = PCA()
    clf_svc = SVC()
    # seta os parametros
    pipe_svc = Pipeline(steps=[("SKB", kb), ("PCA", pca), ("SVC", clf_svc)])

    svc_k = {"SKB__k": range(8, 10)}
    svc_pca = {"PCA__n_components": range(3, 8), "PCA__whiten": [True, False]}
    svc_params = {'SVC__C': [1000],
                  'SVC__gamma': [0.001],
                  'SVC__kernel': ['rbf']}
    svc_k.update(svc_params)
    svc_k.update(svc_pca)
    grid_search = GridSearchCV(pipe_svc,
                               svc_k,
                               n_jobs=-1,
                               cv=cv,
                               scoring='f1')
    grid_search.fit(features_train, labels_train)

    p5aux.imprime_resultado(tabela_resultado,
                            grid_search,
                            svc_k,
                            "Suport Vector Classifier")

def classificador_decision_tree():

    kb = SelectKBest()
    pca = PCA()
    clf_dt = DecisionTreeClassifier()

    pipe = Pipeline(steps=[("SKB", kb),
                           ("PCA", pca),
                           ("DecisionTreeClassifier",
                           clf_dt)])

    dt_k = {"SKB__k": range(8, 10)}
    dt_params = {"DecisionTreeClassifier__min_samples_leaf": [2, 6, 10, 12],
                 "DecisionTreeClassifier__min_samples_split": [2, 6, 10, 12],
                 "DecisionTreeClassifier__criterion": ["entropy", "gini"],
                 "DecisionTreeClassifier__max_depth": [None, 5],
                 "DecisionTreeClassifier__random_state": [42, 46, 60]}
    dt_pca = {"PCA__n_components": range(4, 7), "PCA__whiten": [True, False]}

    dt_k.update(dt_params)
    dt_k.update(dt_pca)

#    enron.get_best_parameters_reports(pipe, dt_k, features, labels)
    grid_search = GridSearchCV(pipe,
                               dt_k,
                               n_jobs=-1,
                               cv=cv,
                               scoring='f1')
    grid_search.fit(features_train, labels_train)
    p5aux.imprime_resultado(tabela_resultado, grid_search, dt_k,"Decision Tree")

def classificador_random_forest():

    kb = SelectKBest()
    clf_rf = RandomForestClassifier()

    pipe_rf = Pipeline(steps=[("SKB", kb), ("RandomForestClassifier", clf_rf)])

    rf_k = {"SKB__k": range(8, 11)}
    rf_params = {'RandomForestClassifier__max_depth': [None, 5, 10],
                  'RandomForestClassifier__n_estimators': [10, 15, 20, 25],
                  'RandomForestClassifier__random_state': [42, 46, 60]}

    rf_k.update(rf_params)

#    enron.get_best_parameters_reports(pipe_rf, rf_k, features, labels)
    grid_search = GridSearchCV(pipe_rf,
                               rf_k,
                               n_jobs=-1,
                               cv=cv,
                               scoring='f1')
    grid_search.fit(features_train, labels_train)

    p5aux.imprime_resultado(tabela_resultado,
                            grid_search,
                            rf_k,
                            "Random Forest")


def classificador_ada_boost():

    kb = SelectKBest()
    clf_ab = AdaBoostClassifier()

    pipe_ab = Pipeline(steps=[("SKB", kb), ("AdaBoostClassifier", clf_ab)])

    ab_k = {"SKB__k": range(8, 11)}
    ab_params = {'AdaBoostClassifier__n_estimators': [10, 20, 30, 40],
                 'AdaBoostClassifier__algorithm': ['SAMME', 'SAMME.R'],
                 'AdaBoostClassifier__learning_rate': [.8, 1, 1.2, 1.5]}

    ab_k.update(ab_params)

    grid_search = GridSearchCV(pipe_ab,
                               ab_k,
                               n_jobs=-1,
                               cv=cv,
                               scoring='f1')
    grid_search.fit(features_train, labels_train)
    p5aux.imprime_resultado(tabela_resultado, grid_search,ab_k,"Ada Boost")

def teste_logistic_regression():

    kbest_features = p5aux.k_best(my_dataset, features_list, 9)
    clf_lr = Pipeline(steps=[('scaler',
                                StandardScaler()),
                                ('pca',
                                PCA(n_components=4, whiten=False)),
                                    ('classifier',
                                    LogisticRegression(tol=0.001,
                                                       C=1e-08,
                                                       penalty='l2',
                                                       random_state=42))])
    lista = testador.test_classifier(clf_lr, my_dataset, kbest_features)

    tabela_teste.add_column ("Logistic Regression",lista)
    return clf_lr, kbest_features

def teste_svc():

    kbest_features = p5aux.k_best(my_dataset, features_list, 9)

    clf_svc = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=5, whiten=True)),
            ('classifier', SVC(C=1000, gamma=.001, kernel='rbf'))])
    lista = testador.test_classifier(clf_svc, my_dataset, kbest_features)
    tabela_teste.add_column ("SVC",lista)

def teste_decision_tree():

    kbest_features = p5aux.k_best(my_dataset, features_list, 9)

    clf_dt = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=4, whiten=True)),
        ('classifier', DecisionTreeClassifier(criterion='gini',
                                              min_samples_leaf=2,
                                              min_samples_split=10,
                                              random_state=60,
                                              max_depth=None))
    ])
    lista = testador.test_classifier(clf_dt, my_dataset, kbest_features)
    tabela_teste.add_column ("Decision Tree", lista)

def teste_random_forrest():

    kbest_features = p5aux.k_best(my_dataset, features_list, 8)

    clf_rf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(max_depth=5,
                                              n_estimators=10,
                                              random_state=60))
    ])
    lista = testador.test_classifier(clf_rf, my_dataset, kbest_features)
    tabela_teste.add_column ("Random Forest",lista)

def teste_ada_boost():

    kbest_features = p5aux.k_best(my_dataset, features_list, 8)

    clf_ab = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', AdaBoostClassifier(learning_rate=1.2,
                                              n_estimators=40,
                                              algorithm='SAMME.R'))
        ])

    lista = testador.test_classifier(clf_ab, my_dataset, kbest_features)
    tabela_teste.add_column ("Ada Boost", lista)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

if __name__ == '__main__':

#    classificador_logistic_regression()
#    classificador_svc()
#    classificador_decision_tree()
#    classificador_random_forest()
#    classificador_ada_boost()
#    print (tabela_resultado)


###  testes]
#    teste_svc()
#    teste_decision_tree()
#    teste_random_forrest()
#    teste_ada_boost()
    teste_logistic_regression()
    print (tabela_teste)
    clf, features_list = teste_logistic_regression()



# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

    dump_classifier_and_data(clf, my_dataset, features_list)
