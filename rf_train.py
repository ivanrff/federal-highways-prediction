# %%
# ===============================================================
#                      IMPORTS
# ===============================================================
import os
import numpy as np
import pandas as pd
from time import time as now
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engine.encoding import RareLabelEncoder


# ===============================================================
#                CARREGAMENTO DOS DADOS
# ===============================================================
def load_datatran():
    df = pd.concat([
        pd.read_csv("data/datatran2024.csv", sep=';', encoding='latin1'),
        pd.read_csv("data/datatran2025.csv", sep=';', encoding='latin1')
    ], axis=0)
    return df


# ===============================================================
#              FUNÇÕES DE PRÉ-PROCESSAMENTO
# ===============================================================
def criar_timestamp(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(
        df['data_inversa'] + ' ' + df['horario'],
        format="%Y-%m-%d %H:%M:%S"
    )
    df.drop(columns=['data_inversa', 'horario'], inplace=True)
    return df


def limpar_nulos(df):
    df = df.copy()
    df = df.dropna(how='any')
    return df


def remover_colunas_irrelevantes(df):
    return df.drop(columns=[
        "mortos", "feridos_leves", "feridos_graves", "ilesos",
        "ignorados", "feridos", "classificacao_acidente",
        "municipio", "delegacia", "regional",
        "tipo_acidente", "causa_acidente",
        "id", "timestamp", "pessoas", "veiculos"
    ])


def processar_km_lat_lon(df):
    df = df.copy()
    for col in ['km', 'latitude', 'longitude']:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='raise')
    return df


def processar_dia_semana(df):
    df = df.copy()

    mapping = {
        'domingo': 0, 'segunda-feira': 1, 'terça-feira': 2,
        'quarta-feira': 3, 'quinta-feira': 4,
        'sexta-feira': 5, 'sábado': 6
    }

    df['dia_numerico'] = df['dia_semana'].str.lower().map(mapping)
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_numerico'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_numerico'] / 7)
    df.drop(columns=['dia_numerico', 'dia_semana'], inplace=True)
    return df


def processar_uso_solo(df):
    df = df.copy()
    df['uso_solo'] = df['uso_solo'].replace({'Sim': 1, 'Não': 0}).astype(int)
    return df


def criar_target(df):
    df = df.copy()
    df['risco_grave'] = (
        (df['mortos'] > 0) | (df['feridos_graves'] > 0)
    ).astype(int)
    return df


def converter_booleans(df):
    df = df.copy()
    for col in df.columns:
        if df[col].nunique() == 2 and set(df[col].unique()) <= {0, 1}:
            df[col] = df[col].astype(bool)
    return df


def preprocess(df):
    df = criar_target(df)
    df = limpar_nulos(df)
    df = remover_colunas_irrelevantes(df)
    df = processar_km_lat_lon(df)
    df = processar_dia_semana(df)
    df = processar_uso_solo(df)
    df = converter_booleans(df)
    return df


# ===============================================================
#                   PIPELINES DE ENCODING
# ===============================================================
class MultiLabelBinarizerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, separator=";"):
        self.separator = separator
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        X_split = X.iloc[:, 0].fillna("").apply(lambda x: x.split(self.separator))
        self.mlb.fit(X_split)
        return self

    def transform(self, X):
        X_split = X.iloc[:, 0].fillna("").apply(lambda x: x.split(self.separator))
        data = self.mlb.transform(X_split)
        return pd.DataFrame(data, columns=self.mlb.classes_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.mlb.classes_


def criar_preprocessor(train_df):

    cat_cols = train_df.select_dtypes(include="object").columns.tolist()
    cat_cols.remove("tracado_via")

    cat_pipeline = Pipeline(steps=[
        ("rare", RareLabelEncoder(tol=0.03, n_categories=1)),
        ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    tracado_pipeline = Pipeline(steps=[
        ("multilabel", MultiLabelBinarizerWrapper(separator=";"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, cat_cols),
            ("tracado", tracado_pipeline, ["tracado_via"])
        ],
        remainder="passthrough"
    )

    return pre


# ===============================================================
#                  BALANCEAMENTO DO TARGET
# ===============================================================
def balancear_dataset(df, y_col):
    true_df = df[df[y_col] == 1]
    false_df = df[df[y_col] == 0].sample(n=len(true_df), random_state=17)
    return pd.concat([true_df, false_df], axis=0)


# ===============================================================
#                    PLOT ROC AUC
# ===============================================================
def plot_roc_auc(y_true, y_score, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    value = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    title = f"{label} (AUC={value:.3f})" if label else f"AUC={value:.3f}"
    plt.plot(fpr, tpr, lw=2, label=title)
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
    plt.legend()
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

    return value


# ===============================================================
#                       EXECUÇÃO PRINCIPAL
# ===============================================================
datatran = load_datatran()
datatran = criar_timestamp(datatran)

# Separar OOT
cut_date = pd.to_datetime("2025-09-01")
oot_df = preprocess(datatran[datatran["timestamp"] >= cut_date])
datatran = preprocess(datatran[datatran["timestamp"] < cut_date])

datatran
# %%

# Train/Test Split
train_df, test_df = train_test_split(
    datatran, test_size=0.1, random_state=17,
    stratify=datatran["risco_grave"]
)

# Pipeline de encoding
preprocessor = criar_preprocessor(train_df)

train_df = pd.DataFrame(
    preprocessor.fit_transform(train_df),
    columns=preprocessor.get_feature_names_out()
)

test_df = pd.DataFrame(
    preprocessor.transform(test_df),
    columns=preprocessor.get_feature_names_out()
)

# Converter colunas numéricas
train_df = train_df.apply(pd.to_numeric)
test_df = test_df.apply(pd.to_numeric)

# Balanceamento
y_col = "remainder__risco_grave"
train_df = balancear_dataset(train_df, y_col)

X_train = train_df.drop(columns=y_col)
y_train = train_df[y_col]

# Treinar modelo
param_grid = {
    "n_estimators": [100, 200],
    "min_samples_leaf": [8, 10, 12],
    "max_depth": [14, 15, 16, None]
}

rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    n_jobs=-1,
    scoring="roc_auc",
    cv=5,
    refit=True
)

t0 = now()
rf.fit(X_train, y_train)
print("Best params:", rf.best_params_)
print(f"Treino em {now() - t0:.1f}s")

# Avaliação ROC
y_score_train = rf.best_estimator_.predict_proba(X_train)[:, 1]
auc_train = plot_roc_auc(y_train, y_score_train, label="Train")

X_test = test_df.drop(columns=y_col)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
y_test = test_df[y_col]

y_score_test = rf.best_estimator_.predict_proba(X_test)[:, 1]
auc_test = plot_roc_auc(y_test, y_score_test, label="Test")

print(f"AUC train = {auc_train:.4f}")
print(f"AUC test  = {auc_test:.4f}")


# cm matrix

y_pred = rf.predict(X_train)

sns.heatmap(confusion_matrix(y_pred=y_pred, y_true=y_train), annot=True, cmap='Blues', fmt='.0f')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

sns.heatmap(confusion_matrix(y_pred=y_pred, y_true=y_train, normalize='true'), annot=True, cmap='Blues', fmt='.0%')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

feature_importances = pd.DataFrame([X_train.columns, rf.best_estimator_.feature_importances_]).T

feature_importances.columns = ['coluna', 'importancia']

feature_importances = feature_importances.sort_values(by='importancia', ascending=False)
# %%
