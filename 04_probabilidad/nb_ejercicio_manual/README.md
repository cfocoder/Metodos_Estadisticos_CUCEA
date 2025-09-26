# Comparación de Implementaciones de Naive Bayes para Clasificación de Spam

Este proyecto compara dos implementaciones del algoritmo Naive Bayes para clasificar mensajes de spam usando datos ficticios: una implementación personalizada y otra usando Scikit-learn.

## Archivos del Proyecto

### `nb_main.py`
Script principal que ejecuta la comparación entre ambas implementaciones:

- **Carga de datos**: Lee el dataset de spam ficticio desde `../../data/spam/spam_ficticio.csv`
- **Preparación de datos de prueba**: Crea un mensaje de prueba codificado como "congratulations, you won free gift" representado como `[1,0,0,0,0,0,0,0]`
- **Implementación personalizada**: Utiliza la función `nb_casero()` del módulo `nb_secondary`
- **Implementación Scikit-learn**: Usa `BernoulliNB` para la misma tarea de clasificación
- **Comparación de resultados**: Muestra las predicciones de ambos métodos

### `nb_secondary.py`
Contiene la implementación personalizada del algoritmo Naive Bayes:

#### Función `nb_casero(df_train, df_test, features, target)`
Implementa el clasificador Naive Bayes desde cero usando el teorema de Bayes:

**Parámetros:**
- `df_train`: DataFrame de entrenamiento
- `df_test`: DataFrame de prueba
- `features`: Lista de variables de entrada
- `target`: Variable objetivo (spam/no spam)

**Proceso:**
1. Calcula tabla de probabilidades (likelihood) agrupando por la variable objetivo
2. Obtiene probabilidades a priori de cada clase
3. Para cada registro de prueba:
   - Calcula P(y=0|x) y P(y=1|x) usando el teorema de Bayes
   - Multiplica probabilidades condicionales para cada característica
   - Asigna la clase con mayor probabilidad

**Retorna:**
- `p_0_list`: Lista de probabilidades P(y=0|x)
- `p_1_list`: Lista de probabilidades P(y=1|x)  
- `pred_list`: Lista de predicciones finales

## Uso

Ejecutar el script principal:
```bash
python nb_main.py
```

El programa mostrará:
- Predicción de la implementación personalizada
- Predicción de BernoulliNB de Scikit-learn

## Objetivo del Ejercicio

Demostrar que ambas implementaciones producen resultados equivalentes, validando la correcta implementación del algoritmo Naive Bayes desde cero y comparándola con una librería establecida.
