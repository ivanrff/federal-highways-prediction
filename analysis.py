# %%

# PROJETO ACIDENTES EM ESTRADAS FEDERAIS

# acidentes2025.csv - cada linha é referente a uma pessoa

# acidentes2025_todas_causas_tipos.csv - semelhante a acidentes2025.csv
# mas com 3 colunas adicionais: 
    # 1. causa_principal (SIM OU NÃO)
        # Provavelmente indica se a causa listada na coluna causa_acidente
        # foi considerada a principal razão para o acidente.
    # 2. ordem_tipo_acidente
        # Quando um acidente envolve múltiplos tipos de eventos (ex: primeiro uma Colisão Traseira e depois um Tombamento)
        
        # este campo provavelmente indica a ordem cronológica ou de relevância dos tipos de acidente (tipo_acidente) 
        # que ocorreram no evento único.
    # 3. Unnamed

# datatran2025.csv - cada linha é referente a um acidente

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)

datatran2025 = pd.read_csv("data/datatran2025.csv", sep=';', encoding='latin1')

# print(f"--\nShape da df: {datatran2025.shape}\n--")

# Buscando por valores vazios
for col in datatran2025.columns:
    n_empty_rows = datatran2025[col].isna().sum()

    # if n_empty_rows > 0:
    #     print(f"Column {col}: {n_empty_rows} empty rows")

# Dropando as poucas linhas que têm valor vazio
datatran2025.dropna(how='any', inplace=True)
# print('[!] Colunas vazias removidas\n--')

# # Checando possíveis valores das colunas categóricas
# cat_cols = datatran2025.select_dtypes(exclude=['int64'])

# # Checando valores em cada coluna categórica
# for col in cat_cols:
#     print(f"Coluna '{col}' valores:")
#     # print("--", datatran2025[col].dtype)
#     print("--", datatran2025[col].unique())

# Criando coluna timestamp a partir de data_inversa e horario
datatran2025['timestamp'] = datatran2025['data_inversa'] + ' ' + datatran2025['horario']
datatran2025['timestamp'] = pd.to_datetime(datatran2025['timestamp'], format="%Y-%m-%d %H:%M:%S")

datatran2025.drop(columns=['data_inversa', 'horario'], inplace=True)

# Algumas colunas numéricas estão como objeto por estarem com , em vez de .
num_cols_with_comma = ['km', 'latitude', 'longitude']

for col in num_cols_with_comma:
    datatran2025[col] = datatran2025[col].apply(lambda x: x.replace(',', '.'))
    datatran2025[col] = datatran2025[col].astype('float64')

# print(datatran2025[num_cols_with_comma].info(), "\n--") # OK

# # Checando possíveis valores das colunas categóricas
# cat_cols = datatran2025.select_dtypes(exclude=['int64', 'float64', 'datetime64[ns]'])

# # Checando valores em cada coluna categórica
# for col in cat_cols:
#     print(f"Coluna '{col}' valores:")
#     print("--", datatran2025[col].dtype)
#     print(datatran2025[col].unique())

# Tratando a coluna tracado_via (transformando com OHE)
tracado_dummies = datatran2025['tracado_via'].str.get_dummies(sep=';')

tracado_dummies.columns = [
        'tracado_' + col.strip()
                        .replace(' ', '_')
                        .lower()
                        .replace('ã', 'a')
                        .replace('á', 'a')
                        .replace('ó', 'o')
                        .replace('ú', 'u')
                        .replace('ç', 'c')
                        .replace('_de_vias', '')
        for col in tracado_dummies.columns
    ]

datatran2025 = pd.concat([datatran2025, tracado_dummies], axis=1)

datatran2025.drop(columns='tracado_via', inplace=True)

# Checando possíveis valores das colunas categóricas
cat_cols = datatran2025.select_dtypes(exclude=['int64', 'float64', 'datetime64[ns]'])

# Checando valores em cada coluna categórica
for col in cat_cols:
    print(f"Coluna '{col}' valores:")
    print("--", datatran2025[col].dtype)
    print(datatran2025[col].unique())
# %%
