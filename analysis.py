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
from sklearn.model_selection import train_test_split

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

train_df, test_df = train_test_split(datatran, stratify=datatran['risco_grave'], test_size=0.1)

def print_distr(y):
    print(f'Distribuição da variável target: {y.sum()/len(y)}')

print('TREINO')
print(train_df.shape)
print_distr(train_df['risco_grave'])
print('TESTE')
print(test_df.shape)
print_distr(test_df['risco_grave'])
print("-------------------------\n")

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

def criar_timestamp(df):
    new_df = df.copy()

    # Criando coluna timestamp a partir de data_inversa e horario
    new_df['timestamp'] = new_df['data_inversa'] + ' ' + new_df['horario']
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], format="%Y-%m-%d %H:%M:%S")

    ## -- DROP --
    new_df.drop(columns=['data_inversa', 'horario'], inplace=True)

    return new_df

train_df = criar_timestamp(train_df)
test_df = criar_timestamp(test_df)

# --------------------------------------------------------------------------------------------
# ----------------------------- COLUNAS km, latitude, longitude ------------------------------
# --------------------------------------------------------------------------------------------

# Algumas colunas numéricas estão como objeto por estarem com , em vez de .
num_cols_with_comma = ['km', 'latitude', 'longitude']

def trocar_virgula_ponto(df, num_cols_with_comma):
    new_df = df.copy()

    for col in num_cols_with_comma:
        new_df[col] = new_df[col].astype(str).str.replace(',', '.', regex=False)
        new_df[col] = pd.to_numeric(new_df[col], errors='raise')

    return new_df

train_df = trocar_virgula_ponto(train_df, num_cols_with_comma)
test_df = trocar_virgula_ponto(test_df, num_cols_with_comma)

# print(train_df[num_cols_with_comma].info(), "\n--") # OK

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

def enc_dia_semana(df, day_mapping):
    new_df = df.copy()

    new_df['dia_numerico'] = new_df['dia_semana'].str.lower().map(day_mapping)

    new_df['dia_semana_sin'] = np.sin(2 * np.pi * new_df['dia_numerico'] / 7)
    new_df['dia_semana_cos'] = np.cos(2 * np.pi * new_df['dia_numerico'] / 7)

    ## -- DROP --
    new_df.drop(columns=['dia_numerico', 'dia_semana'], axis=1, inplace=True)

    return new_df

train_df = enc_dia_semana(train_df, day_mapping)
test_df = enc_dia_semana(test_df, day_mapping)

# --------------------------------------------------------------------------------------------
# ------------------------------------- COLUNA uso_solo --------------------------------------
# --------------------------------------------------------------------------------------------

# Tratamento de uso_solo
def enc_uso_solo(df):
    new_df = df.copy()
    new_df['uso_solo'] = new_df['uso_solo'].replace({'Sim': 1, 'Não': 0}).astype('int64')

    return new_df

train_df = enc_uso_solo(train_df)
test_df = enc_uso_solo(test_df)

# train_df['uso_solo'].value_counts(normalize=True)

# --------------------------------------------------------------------------------------------
# ---------------------------------- COLUNA tracado_via --------------------------------------
# --------------------------------------------------------------------------------------------

train_df['tracado_via'] = train_df['tracado_via'].apply(lambda x: x.strip()
                                                                    .replace(' ', '_')
                                                                    .lower()
                                                                    .replace('ã', 'a')
                                                                    .replace('á', 'a')
                                                                    .replace('ó', 'o')
                                                                    .replace('ú', 'u')
                                                                    .replace('ç', 'c')
                                                                    .replace('_de_vias', ''))

# # Tratando a coluna tracado_via (transformando com OHE)
# tracado_dummies = train_df['tracado_via'].str.get_dummies(sep=';')

# train_df = pd.concat([train_df, tracado_dummies], axis=1)

# ## -- DROP --
# train_df.drop(columns='tracado_via', inplace=True)

# Tratando a coluna tracado_via (transformando com OHE)
tracado_dummies = train_df['tracado_via'].str.get_dummies(sep=';')

train_df = pd.concat([train_df, tracado_dummies], axis=1)

cols_to_drop = []
for col in tracado_dummies.columns:
    # print('------', col)
    rate = train_df[col].sum()/train_df.shape[0]
    # print(rate)
    if rate < 0.05:
        cols_to_drop.append(col)

# print(cols_to_drop)
# ['desvio_temporario', 'em_obras', 'ponte', 'retorno_regulamentado', 'rotatoria', 'tunel', 'viaduto']

train_df.drop(columns=cols_to_drop, inplace=True)


# --------------------------------------------------------------------------------------------
# ------------------- COLUNAS municipio, delegacia, regional, uop, uf ------------------------
# --------------------------------------------------------------------------------------------

# Remover municipio (muita granularidade)
# uop vs. delegacia vs. regional
# Escolher uma para OHE e remover as outras
## -- DROP --
train_df.drop(columns=['municipio', 'delegacia', 'regional'], axis=1, inplace=True)

# uop_regional_dummies = pd.get_dummies(train_df[['uop', 'uf']])
# train_df = pd.concat([train_df, uop_regional_dummies], axis=1)

# ## -- DROP --
# train_df.drop(columns=['uop', 'uf'], axis=1, inplace=True)

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

# Checando possíveis valores das colunas categóricas
cat_cols = train_df.select_dtypes(include=['object']).columns

print("-----------------------------------------")
print("Colunas categóricas:\n", cat_cols)
print("-----------------------------------------")

# Checando valores em cada coluna categórica
for n, col in enumerate(cat_cols):
    print(f"\n{n+1}. Coluna '{col}' valores:")
    print("--", train_df[col].dtype)
    # print(train_df[col].unique())
    print(train_df[col].value_counts(normalize=True))
# %%
