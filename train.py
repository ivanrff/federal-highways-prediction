
# %%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

datatran = pd.concat([
    pd.read_csv("data/datatran2024.csv", sep=';', encoding='latin1'),
    pd.read_csv("data/datatran2025.csv", sep=';', encoding='latin1')
], axis=0)

def criar_timestamp(df):
    new_df = df.copy()

    # Criando coluna timestamp a partir de data_inversa e horario
    new_df['timestamp'] = new_df['data_inversa'] + ' ' + new_df['horario']
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], format="%Y-%m-%d %H:%M:%S")

    ## -- DROP --
    new_df.drop(columns=['data_inversa', 'horario'], inplace=True)

    return new_df

def preprocess(df):
    new_df = df.copy()

    # --------------------------------------------------------------------------------------------
    # --------------------------------------- DADOS NULOS ----------------------------------------
    # --------------------------------------------------------------------------------------------

    for col in new_df.columns:
        n_empty_rows = new_df[col].isna().sum()

        if n_empty_rows > 0:
            print(f"Coluna {col}: {n_empty_rows} linhas nulas")

    # Dropando as poucas linhas que têm valor vazio
    new_df.dropna(how='any', inplace=True)
    print(f"Número de linhas com valores nulos removidas: {df.shape[0] - new_df.shape[0]}")

    # --------------------------------------------------------------------------------------------
    # ---------------------------------- DADOS DUPLICADOS ----------------------------------------
    # --------------------------------------------------------------------------------------------

    print(f'{new_df.duplicated().sum()} linhas duplicadas\n')

    # --------------------------------------------------------------------------------------------
    # -------------------------------------- COLUNA TARGET ---------------------------------------
    # --------------------------------------------------------------------------------------------

    new_df['risco_grave'] = ((new_df['mortos'] > 0) | (new_df['feridos_graves'] > 0)).astype(int)

    new_df.drop(columns=["mortos", "feridos_leves", "feridos_graves", "ilesos", "ignorados", "feridos",
                        "classificacao_acidente"], inplace=True)

    # --------------------------------------------------------------------------------------------
    # ----------------------------- COLUNAS km, latitude, longitude ------------------------------
    # --------------------------------------------------------------------------------------------

    # Algumas colunas numéricas estão como objeto por estarem com , em vez de .
    num_cols_with_comma = ['km', 'latitude', 'longitude']

    for col in num_cols_with_comma:
        new_df[col] = new_df[col].astype(str).str.replace(',', '.', regex=False)
        new_df[col] = pd.to_numeric(new_df[col], errors='raise')

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

    new_df['dia_numerico'] = new_df['dia_semana'].str.lower().map(day_mapping)

    new_df['dia_semana_sin'] = np.sin(2 * np.pi * new_df['dia_numerico'] / 7)
    new_df['dia_semana_cos'] = np.cos(2 * np.pi * new_df['dia_numerico'] / 7)

    ## -- DROP --
    new_df.drop(columns=['dia_numerico', 'dia_semana'], axis=1, inplace=True)

    # --------------------------------------------------------------------------------------------
    # ------------------------------------- COLUNA uso_solo --------------------------------------
    # --------------------------------------------------------------------------------------------

    new_df['uso_solo'] = (new_df['uso_solo']
                            .replace({'Sim': '1', 'Não': '0'}) # coloquei '1' e '0' para suprimir o Warning de downcasting do pandas
                                .astype('int64'))

    # --------------------------------------------------------------------------------------------
    # ------------------- COLUNAS municipio, delegacia, regional, uop, uf ------------------------
    # --------------------------------------------------------------------------------------------

    # Remover municipio (muita granularidade)
    # uop vs. delegacia vs. regional
    # Escolher uma para OHE e remover as outras
    ## -- DROP --
    new_df.drop(columns=['municipio', 'delegacia', 'regional'], axis=1, inplace=True)

    # --------------------------------------------------------------------------------------------
    # -------------------------- COLUNAS tipo_acidente e causa_acidente --------------------------
    # --------------------------------------------------------------------------------------------

    # remover tipo_acidente e causa_acidente (informação não conhecida no momento da predição)
    new_df.drop(columns=['tipo_acidente', 'causa_acidente'], axis=1, inplace=True)
    
    # --------------------------------------------------------------------------------------------
    # ---------------------------------- COLUNAS id e timestamp ----------------------------------
    # --------------------------------------------------------------------------------------------

    new_df.drop(columns=['id', 'timestamp'], inplace=True)

    # --------------------------------------------------------------------------------------------
    # ------------------------------------ COLUNAS booleanas -------------------------------------
    # --------------------------------------------------------------------------------------------

    # transformar as colunas booleanas em tipo bool
    for col in new_df.columns:
        if new_df[col].max() == 1 \
            and new_df[col].min() == 0 \
            and new_df[col].nunique() == 2:
                new_df[col] = new_df[col].astype('bool')

    return new_df

datatran = criar_timestamp(datatran)

oot_df = datatran[datatran['timestamp'] >= pd.to_datetime('2025-09-01 00:00:00')]
datatran = datatran[datatran['timestamp'] < pd.to_datetime('2025-09-01 00:00:00')]

oot_df = preprocess(oot_df)
datatran = preprocess(datatran)

train_df, test_df = train_test_split(datatran, test_size=0.1, random_state=17, stratify=datatran['risco_grave'])
# %%