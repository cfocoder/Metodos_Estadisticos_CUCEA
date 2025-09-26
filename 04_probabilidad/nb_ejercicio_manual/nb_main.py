#!/opt/homebrew/Caskroom/miniconda/base/envs/ds_env/bin/python
# -*- coding: utf-8 -*-
# Author: Javier Pérez

# Importación de librerías
import numpy as np
import pandas as pd
from nb_secondary import nb_casero # importar una función a partir de un script externo
from sklearn.naive_bayes import BernoulliNB

def main():

    df_spam = pd.read_csv("../../data/spam/spam_ficticio.csv")
    #print(df_spam.head(20))

    # Lista de variables de entrada y objetivo
    target = 'spam'
    features = df_spam.columns.drop(target)

    # Dataframe de prueba
    df_test = pd.DataFrame(data={},
                       columns=features)
    # Añadimos un renglón al set de validacion
    df_test.loc[len(df_test.index)] = [1,0,0,0,0,0,0,0] # codificacion de mensaje "congratulations, you won free gift"

    #print(df_test.head())

    p_0_list, p_1_list, pred_list = nb_casero(df_train=df_spam,
                                              df_test=df_test,
                                              features=features,
                                              target=target)
    #print('p_0: ', p_0_list)
    #print('p_1: ', p_1_list)
    print('clase predicha manualmente: ', pred_list)

    # 1. Importar la librería
    # 2. Instanciar el modelo
    model = BernoulliNB()
    # 3. X->matriz de características, y->vector objetivo
    X_train = df_spam[features]
    y_train = df_spam[target]
    # 4. Entrenar el modelo
    model.fit(X_train, y_train)
    # 5. Predicciones
    X_test = df_test.copy()
    print('Clase predicha por BernoulliNB: ', model.predict(X_test))



if __name__ == "__main__":
    main()