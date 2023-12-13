import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
import warnings
import os
os.getcwd()
warnings.filterwarnings("ignore")

#### Módulo de Probabilidad de Riesgo

def load_df():
    
    rEmbarques = './data/Salidas Mondelez.xlsx'
    
    Embarques = pd.read_excel(rEmbarques, sheet_name = "Data")

    Embarques['Inicio'] = pd.to_datetime(Embarques['Inicio'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    Embarques['Arribo'] = pd.to_datetime(Embarques['Arribo'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    Embarques['Finalización'] = pd.to_datetime(Embarques['Finalización'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    Embarques.Arribo.fillna(Embarques.Finalización, inplace=True)
    Embarques['TiempoCierreServicio'] = (Embarques['Finalización'] - Embarques['Arribo'])
    Embarques['TiempoCierreServicio'] = Embarques['TiempoCierreServicio']/np.timedelta64(1,'h')
    Embarques['TiempoCierreServicio'].fillna(Embarques['TiempoCierreServicio'].mean(), inplace=True)
    Embarques['TiempoCierreServicio'] = Embarques['TiempoCierreServicio'].astype(int)

    Embarques['Destinos'].fillna('OTRO', inplace=True)
    Embarques['Línea Transportista'].fillna('OTRO', inplace=True)
    Embarques['Duración'].fillna(Embarques['Duración'].mean(), inplace=True)
    Embarques['Duración'] = Embarques['Duración'].astype(int)
    #Embarques['Año'] = Embarques['Inicio'].apply(lambda x: x.year)
    #Embarques['Hora'] = Embarques['Inicio'].apply(lambda x: x.hour)
    Embarques['Mes'] = Embarques['Inicio'].apply(lambda x: x.month)
    Embarques['DiadelAño'] = Embarques['Inicio'].apply(lambda x: x.dayofyear)
    Embarques['SemanadelAño'] = Embarques['Inicio'].apply(lambda x: x.weekofyear)
    Embarques['DiadeSemana'] = Embarques['Inicio'].apply(lambda x: x.dayofweek)
    Embarques['Quincena'] = Embarques['Inicio'].apply(lambda x: x.quarter)
    #Embarques['Mes'] = Embarques['MesN'].map({1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"})
    #Embarques['Dia'] = Embarques['Inicio'].apply(lambda x: x.day)
    Embarques['Origen Destino'] = Embarques['Estado Origen'] + '-' + Embarques['Estado Destino']
    Embarques = Embarques[['Origen Destino','Tipo Monitoreo', 'Tipo Unidad', 'Duración', 'Mes', 'DiadelAño', 'SemanadelAño', 'DiadeSemana', 'Quincena', 'Robo']]
    Embarques = Embarques.dropna()
    #Embarques = Embarques.dropna() 
    return Embarques

df = load_df()

#df.drop(['Bitácora','Cliente','Origen','Estado Origen','Destinos','Estado Destino','Línea Transportista','Inicio','Arribo','Finalización','Tiempo Recorrido'], axis = 'columns', inplace=True)

target = 'Robo'
encode = ['Origen Destino', 'Tipo Unidad', 'Tipo Monitoreo']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'NO':0, 'SI':1}
def target_encode(val):
    return target_mapper[val]

df['Robo'] = df['Robo'].apply(target_encode)

# Separating X and y
X = df.drop('Robo', axis=1)
Y = df['Robo']

# Build random forest model

dfSI = len(df.loc[df.loc[:, 'Robo'] == 1])
dfNO = len(df.loc[df.loc[:, 'Robo'] == 0])
cw = dfNO/dfSI

clf = DecisionTreeClassifier(criterion='entropy',
                                            min_samples_split=20,
                                            min_samples_leaf=5,
                                            max_depth = 4,
                                            class_weight={1:cw})
clf.fit(X, Y)

# Saving the model
pickle.dump(clf, open(r'proba_robo_mondelez.pkl', 'wb'))
