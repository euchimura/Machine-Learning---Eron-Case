#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import csv
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

import pandas as pd


## Remove os outliers dos dados
def remove_outliers(data_dict):
    data_dict.pop('TOTAL',0)
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)


#Criação de novos features - 'por_from_poi' e 'por_to_poi'
# retorna os dados com as novas features
def cria_atributos(data_dict):
    for nome in data_dict:
    #por_from_poi
        emails_recebidos = data_dict[nome]['from_poi_to_this_person']
        tot_emails_recebidos = data_dict[nome]['to_messages']

        if emails_recebidos == 'NaN' or tot_emails_recebidos =='NaN':
            data_dict[nome]['por_from_poi'] = 0.0
        else:
            data_dict[nome]['por_from_poi'] = float(emails_recebidos)/float(tot_emails_recebidos)

        #por_to_poi
        emails_enviados = data_dict[nome]['from_this_person_to_poi']
        tot_emails_enviados = data_dict[nome]['from_messages']

        if emails_enviados == 'NaN' or tot_emails_enviados =='NaN':
            data_dict[nome]['por_to_poi'] = 0.0
        else:
            data_dict[nome]['por_to_poi'] = float(emails_enviados)/float(tot_emails_enviados)

    return data_dict

# lista as features retirando as features que possuem mais que 50% de NAN
# retorna uma lista de featuers com o 'poi' no inicio da lista
def inicia_features ():
    features_list = ['poi',
                     'bonus',
    ##                 'deferral_payments', ## retirado
    ##                 'deferred_income', ## retirado
    ##                 'director_fees', retirado
                     'exercised_stock_options',
                     'expenses',
    ##                 'loan_advances', ## retirado
    ##                 'long_term_incentive', ##retirado
                     'other',
                     'restricted_stock',
    ##                 'restricted_stock_deferred', ## retirado
                     'salary',
                     'total_payments',
                     'total_stock_value',
                     'from_messages',
                     'from_poi_to_this_person',
                     'from_this_person_to_poi',
                     'shared_receipt_with_poi',
                     'to_messages',
                     'por_to_poi',
                     'por_from_poi']

    return features_list
## inicia as features sem as features criadas: por_to_poi e por_from_poi
def inicia_features_limpo ():

    features_list = ['poi',
                     'bonus',
    ##                 'deferral_payments', ## retirado
    ##                 'deferred_income', ## retirado
    ##                 'director_fees', retirado
                     'exercised_stock_options',
                     'expenses',
    ##                 'loan_advances', ## retirado
    ##                 'long_term_incentive', ##retirado
                     'other',
                     'restricted_stock',
    ##                 'restricted_stock_deferred', ## retirado
                     'salary',
                     'total_payments',
                     'total_stock_value',
                     'from_this_person_to_poi',
                     'from_messages',
                     'from_poi_to_this_person',
                     'shared_receipt_with_poi',
                     'to_messages']
    return features_list
# retorna os k melhores features dado de entrada um data set,
# lista de features e numero de k.

def k_best(data_dict, features_list, k):

    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    kbest = SelectKBest(f_classif, k=k)
    kbest.fit(features, labels)

    feature_list = zip(features_list[1:], kbest.scores_)
    ordenado = sorted(feature_list, key=lambda x: x[1], reverse=True)
    k_best_features = dict(ordenado[:k])
    return ['poi'] + k_best_features.keys()

#funcao que imprime o resultado do gridsearch
# em uma tabela prettytable
import prettytable
from prettytable import ALL as ALL

def imprime_resultado(tabela, gridsearch, parametros, nome):
    pars = ""
    melhor_resultado = gridsearch.best_estimator_.get_params()
    for nome_parametro in sorted(parametros.keys()):
        pars = pars + nome_parametro + ': ' + str(melhor_resultado[nome_parametro]) +'\n'


    tabela.add_row([nome, gridsearch.best_score_, pars])

# inicia uma tabela de resultado - prettytable
def inicia_resultado():
    tab = prettytable.PrettyTable(hrules=ALL)
    tab.field_names = ['Classificador', 'Pontuacao', 'Atributos']
    return tab

def inicia_tabela_testes():
    tab = prettytable.PrettyTable(hrules=ALL)
    tab.add_column ("Classificador",['Accuracy', 'Precision', 'Recal', 'f1', 'f2', 'Total Predictions', 'true_positives','false Positives', 'false Negatives', 'true_negatives'])
    return tab

# inicia a tabela panda para testes
def inicia_pandas_testes():

    nomes = ['Classificador',
             'k',
             'Accuracy',
             'Precision',
             'Recal',
             'f1',
             'f2',
             'Total Predictions',
             'true_positives',
             'false Positives',
             'false Negatives',
             'true_negatives']
    tab_panda = pd.DataFrame(columns = nomes)
    return tab_panda

# insere na tabela os valores
def insere_panda(tabela, nome, listadados,k):
    a = listadados
    a.insert(0,k)
    a.insert(0,nome)
    tabela.loc[len(tabela.index)+1] = a
    return tabela
