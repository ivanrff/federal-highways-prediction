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
from feature_engine.encoding import RareLabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Set max rows to display all rows
pd.set_option('display.max_rows', 30)

pd.set_option('display.max_columns', 20)

datatran2025 = pd.read_csv("data/datatran2025.csv", sep=';', encoding='latin1')
datatran2024 = pd.read_csv("data/datatran2024.csv", sep=';', encoding='latin1')

datatran = pd.concat([datatran2025, datatran2024], axis=0)

del datatran2024, datatran2025

# print(f"--\nShape da df: {datatran.shape}\n--")
# for col in datatran.columns:
#     print(col)

# --------------------------------------------------------------------------------------------
# --------------------------------------- DADOS NULOS ----------------------------------------
# --------------------------------------------------------------------------------------------

# Buscando por valores vazios
for col in datatran.columns:
    n_empty_rows = datatran[col].isna().sum()

    # if n_empty_rows > 0:
    #     print(f"Column {col}: {n_empty_rows} empty rows")

# Dropando as poucas linhas que têm valor vazio
datatran.dropna(how='any', inplace=True)
# print('[!] Colunas vazias removidas\n--')

# --------------------------------------------------------------------------------------------
# ---------------------------------- DADOS DUPLICADOS ----------------------------------------
# --------------------------------------------------------------------------------------------

# print(f'{datatran.duplicated().sum()} linhas duplicadas.\n')

# --------------------------------------------------------------------------------------------
# -------------------------------------- COLUNA TARGET ---------------------------------------
# --------------------------------------------------------------------------------------------

datatran['risco_grave'] = ((datatran['mortos'] > 0) | (datatran['feridos_graves'] > 0)).astype(int)

# - `pessoas`: número de pessoas envolvidas
# - `mortos`: número de mortos
# - `feridos_leves`: número de feridos leves
# - `feridos_graves`: número de feridos graves
# - `ilesos`: número de ilesos
# - `ignorados`: total de pessoas envolvidas na ocorrência e que não se soube o estado físico.
# - `feridos`: total de feridos
# - `veiculos`: veículos envolvidos

datatran.drop(columns=["mortos", "feridos_leves", "feridos_graves", "ilesos", "ignorados", "feridos",
                       "classificacao_acidente"], inplace=True)

# --------------------------------------------------------------------------------------------
# ------------------------------ TRATAMENTO COLUNAS CATEGÓRICAS ------------------------------
# --------------------------------------------------------------------------------------------

# # Checando possíveis valores das colunas categóricas
# cat_cols = datatran.select_dtypes(include=['object']).columns

# # Checando valores em cada coluna categórica
# for col in cat_cols:
#     print(f"Coluna '{col}' valores:")
#     # print("--", datatran[col].dtype)
#     print("--", datatran[col].unique())

# --------------------------------------------------------------------------------------------
# ------------------------------ COLUNAS data_inversa, horario -------------------------------
# --------------------------------------------------------------------------------------------

# Criando coluna timestamp a partir de data_inversa e horario
datatran['timestamp'] = datatran['data_inversa'] + ' ' + datatran['horario']
datatran['timestamp'] = pd.to_datetime(datatran['timestamp'], format="%Y-%m-%d %H:%M:%S")


## -- DROP --
datatran.drop(columns=['data_inversa', 'horario'], inplace=True)


# --------------------------------------------------------------------------------------------
# ----------------------------- COLUNAS km, latitude, longitude ------------------------------
# --------------------------------------------------------------------------------------------

# Algumas colunas numéricas estão como objeto por estarem com , em vez de .
num_cols_with_comma = ['km', 'latitude', 'longitude']

for col in num_cols_with_comma:
    datatran[col] = datatran[col].astype(str).str.replace(',', '.', regex=False)
    datatran[col] = pd.to_numeric(datatran[col], errors='raise')

# print(datatran[num_cols_with_comma].info(), "\n--") # OK


# --------------------------------------------------------------------------------------------
# ---------------------------------- COLUNA dia_semana ---------------------------------------
# --------------------------------------------------------------------------------------------

# mapeando os dias da semana de forma cíclica
day_mapping = {
    'domingo': 0,
    'segunda-feira': 1,
    'terça-feira': 2,
    'quarta-feira': 3,
    'quinta-feira': 4,
    'sexta-feira': 5,
    'sábado': 6
}
datatran['dia_numerico'] = datatran['dia_semana'].str.lower().map(day_mapping)

datatran['dia_semana_sin'] = np.sin(2 * np.pi * datatran['dia_numerico'] / 7)
datatran['dia_semana_cos'] = np.cos(2 * np.pi * datatran['dia_numerico'] / 7)

## -- DROP --
datatran.drop(columns=['dia_numerico', 'dia_semana'], axis=1, inplace=True)


# --------------------------------------------------------------------------------------------
# ------------------------------------- COLUNA uso_solo --------------------------------------
# --------------------------------------------------------------------------------------------

# Tratamento de uso_solo
datatran['uso_solo'] = datatran['uso_solo'].replace({'Sim': 1, 'Não': 0}).astype('int64')
# datatran['uso_solo'].value_counts(normalize=True) # ({0: 0.569314, 1: 0.430686})

# --------------------------------------------------------------------------------------------
# ---------------------------------- COLUNA tracado_via --------------------------------------
# --------------------------------------------------------------------------------------------

datatran['tracado_via'] = datatran['tracado_via'].apply(lambda x: x.strip()
                                                                    .replace(' ', '_')
                                                                    .lower()
                                                                    .replace('ã', 'a')
                                                                    .replace('á', 'a')
                                                                    .replace('ó', 'o')
                                                                    .replace('ú', 'u')
                                                                    .replace('ç', 'c')
                                                                    .replace('_de_vias', ''))

# # Tratando a coluna tracado_via (transformando com OHE)
# tracado_dummies = datatran['tracado_via'].str.get_dummies(sep=';')

# datatran = pd.concat([datatran, tracado_dummies], axis=1)

# ## -- DROP --
# datatran.drop(columns='tracado_via', inplace=True)

# Tratando a coluna tracado_via (transformando com OHE)
tracado_dummies = datatran['tracado_via'].str.get_dummies(sep=';')

datatran_tv = pd.concat([datatran[['tracado_via']], tracado_dummies], axis=1)

# --------------------------------------------------------------------------------------------
# ------------------- COLUNAS municipio, delegacia, regional, uop, uf ------------------------
# --------------------------------------------------------------------------------------------

# Remover municipio (muita granularidade)
# uop vs. delegacia vs. regional
# Escolher uma para OHE e remover as outras
## -- DROP --
datatran.drop(columns=['municipio', 'delegacia', 'regional'], axis=1, inplace=True)

# uop_regional_dummies = pd.get_dummies(datatran[['uop', 'uf']])
# datatran = pd.concat([datatran, uop_regional_dummies], axis=1)

# ## -- DROP --
# datatran.drop(columns=['uop', 'uf'], axis=1, inplace=True)

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

# Checando possíveis valores das colunas categóricas
cat_cols = datatran.select_dtypes(include=['object']).columns

print("-----------------------------------------")
print("Colunas categóricas:\n", cat_cols)
print("-----------------------------------------")

# Checando valores em cada coluna categórica
for n, col in enumerate(cat_cols):
    print(f"\n{n+1}. Coluna '{col}' valores:")
    print("--", datatran[col].dtype)
    # print(datatran[col].unique())
    print(datatran[col].value_counts(normalize=True))
# %%
