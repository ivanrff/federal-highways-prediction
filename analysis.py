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

pd.set_option('display.max_columns', 500)

datatran2025 = pd.read_csv("data/datatran2025.csv", sep=';', encoding='latin1')

# Buscando por valores vazios
for col in datatran2025.columns:
    n_empty_rows = datatran2025[col].isna().sum()

    if n_empty_rows > 0:
        print(f"Column {col}: {n_empty_rows} empty rows")

# Checando possíveis valores das colunas categóricas
cat_cols = datatran2025.select_dtypes(exclude=['int64'])

# Checando valores em cada coluna categórica
for col in cat_cols:
    print(col)
    print("--", datatran2025[col].dtype)
    print("--", datatran2025[col].unique())

# Criando coluna timestamp a partir de data_inversa e horario
datatran2025['timestamp'] = datatran2025['data_inversa'] + ' ' + datatran2025['horario']
datatran2025['timestamp'] = pd.to_datetime(datatran2025['timestamp'], format="%Y-%m-%d %H:%M:%S")

# %%
