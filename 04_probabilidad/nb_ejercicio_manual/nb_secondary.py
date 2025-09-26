def nb_casero(df_train,df_test,features,target):
    '''
    Función custom para hacer la clasificación en base al teorema de Bayes

    Parámetros de entrada

    df_train:  DataFrame de entrenamiento
    df_test: DataFrame de prueba
    features: str, columnas/variables de entrada
    target: str, variable dependiente/objetivo

    Return:
    p_0_list: P(y=0|x)
    p_1_list: P(y=1|x)
    pred_list: Clase
    '''
    import numpy as np
    import pandas as pd
    
    # tabla de probabilidad(likelihood)
    df_prob = df_train.groupby(target).mean()
    
    p_0_list = []
    p_1_list = []
    pred_list=[] # lista de predicciones

    # Probabilidades totales del target
    p_target_0 = df_train.groupby(target).size()[0]/df_train.shape[0]
    p_target_1 = df_train.groupby(target).size()[1]/df_train.shape[0]

    # iteramos sobre todos los registros(filas) utilizando método iterrows()
    for index, row in df_test.iterrows():
        
        p_0 = p_target_0 # inicializamos la probabilidad condicional total
        p_1 = p_target_1 # inicializamos la probabilidad condicional total

        #iteracion sobre las variables
        for var in features:
            if row[var]==1: # caso var=1
                p_0 *= df_prob[var].loc[0] # se multiplica por su respectiva entrada en la tabla de prob.
                p_1 *= df_prob[var].loc[1]
            elif row[var]==0: # caso var=0
                p_0 *= 1.0-df_prob[var].loc[0] # se multiplica por su respectivo complemento en la tabla de prob.
                p_1 *= 1.0-df_prob[var].loc[1]
        
        p_0_list.append(p_0)
        p_1_list.append(p_1)
        pred = np.where(p_1>p_0,1,0) # Se comparan las dos prob condicionales, se elige la etiqueta de la mayor
        pred_list.append(int(pred)) # Se agrega a la lista de predicciones
        #print('p_0=',round(p_0,5),'p_1=',round(p_1,5),'pred=',int(pred))
        
    return p_0_list, p_1_list, pred_list

if __name__ == '__main__':
    print(__name__)