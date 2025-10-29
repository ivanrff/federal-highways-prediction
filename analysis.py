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

# %%
import pandas as pd

pd.set_option('display.max_columns', 500)

datatran2025 = pd.read_csv("data/datatran2025.csv", sep=';', encoding='latin1')

print(datatran2025.columns)

datatran2025.head(9)
# %%
