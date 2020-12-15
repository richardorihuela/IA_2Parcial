"""
Author: Orihuela Gil, Richard Hector
"""


#*************************************************CODIGO*************************************************
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import numpy as np
import matplotlib.pyplot as pp
import random

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import pandas as pd
import csv
from sklearn.impute import SimpleImputer

#imputacion = SimpleImputer(missing_values=np.nan, strategy="mean")

#leemos archivo
data = pd.read_csv("crx.csv")

print(data)


#<DATA CLEANING>
#contar valores perdidos en columna
print("*****Datos Faltantes*****")
missing_values_count = data.isnull().sum()
print(missing_values_count)
#print("*****Datos Faltantes*****")
#print(data.isnull().sum(axis=0))
#>>>Tenemos datos faltantes<<<

#¿Cuántos valores perdidos totales tenemos?
total_cells = np.product(data.shape)
total_missing = missing_values_count.sum()

#porcentaje de datos que faltan
percent_missing = (total_missing/total_cells) * 100
print("Porcentaje de datos faltante")
print(percent_missing)

#Eliminar las filas con valores faltantes
#print("SIN VALORES FALTANTES")
#print(data.dropna ())

#Reemplazar con el valor siguiente
# reemplace todos los NaN por el valor que viene directamente después en la misma columna,
# luego reemplace todas las na restantes con 0
data = data.fillna(method='bfill', axis=0).fillna(0)



#Reemplazar + por 1 y - por 0
dataSinSigno = data.replace({"+": 1, "-": 0})   #trabajaremos con estos datos

#</DATA CLEANING>

print("--------------------------------MOSTRAR--------------------------------")

print("*********X*********")
#separar la columna signo
#X = dataSinSigno.iloc[:, 0:14]
#print(X)

#columnas con valores numericos: 1, 2, 7, 10, 13, 14
X = dataSinSigno.iloc[:,[1,2,7,10,13,14]]
print(X)

print("*********y*********")
#sacar la columna signo
y = dataSinSigno.iloc[:, 15]
print(y)

#>>>Entrenar<<<

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

x_train, x_test, d_train, d_test = train_test_split(X, y, test_size=0.80, random_state=42)
#"Lbfgs" es un optimizador de la familia de métodos cuasi-Newton.
#"Logistic", la función sigmoidea logística, devuelve f (x) = 1 / (1 + exp (-x)).
#"verbose", imprimir mensajes de progreso en stdout.
#"alpha", L2 penalty (regularization term) parameter.
#"hidden_layer_sizes", El i-ésimo elemento representa el número de neuronas en la i-ésima capa oculta.
mlp=MLPClassifier(solver = 'lbfgs', activation='logistic', verbose=True, alpha=1e-4, tol = 1e-3, hidden_layer_sizes=(150, 2))
#La clase MLPClassifier implementa un algoritmo de perceptrón multicapa (MLP) que entrena mediante la retropropagación.
#MLP se entrena en dos arreglos:
#   arreglo X de tamaño (n_samples, n_features),
#       que contiene las muestras de entrenamiento representadas como vectores de características de punto flotante
#   matriz y de tamaño (n_samples,), que contiene los valores objetivo (etiquetas de clase) para las muestras de entrenamiento:


mlp.fit(X, y)
prediccion = mlp.predict(x_test)
print('Matriz de Confusion\n')
matriz = confusion_matrix(d_test, prediccion)
print(confusion_matrix(d_test, prediccion))
print('\n')
print(classification_report(d_test, prediccion))
#print(X, y)
















