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
from tqdm import tqdm
import warnings
from sklearn.utils import resample

warnings.filterwarnings(
    "ignore", 
    message="This Pipeline instance is not fitted yet.", 
    category=FutureWarning
)


# ===============================================================
#                CARREGAMENTO DOS DADOS
# ===============================================================
def load_datatran():
    df = pd.concat([
        pd.read_csv("data/datatran2023.csv", sep=';', encoding='latin1'),
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

def criar_colunas_tempo(df):
    df = df.copy()

    # Mês
    df['mes'] = df['timestamp'].dt.month

    # Hora do dia em formato cíclico
    df['hora'] = df['timestamp'].dt.hour
    df['minuto'] = df['timestamp'].dt.minute
    df['segundo'] = df['timestamp'].dt.second

    df['fracao_dia'] = (df['hora'] + df['minuto']/60 + df['segundo']/3600) /24

    df['hora_sin'] = np.sin(2 * np.pi * df['fracao_dia'])
    df['hora_cos'] = np.cos(2 * np.pi * df['fracao_dia'])

    df.drop(columns=['hora', 'minuto', 'segundo', 'fracao_dia'], inplace=True)

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
        "id", "timestamp", "veiculos"
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
    df['uso_solo'] = df['uso_solo'].replace({'Sim': '1', 'Não': '0'}).astype(int)
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
    df = criar_colunas_tempo(df)
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


from sklearn.preprocessing import StandardScaler

def criar_preprocessor(train_df):

    cat_cols = train_df.select_dtypes(include="object").columns.tolist()
    cat_cols.remove("tracado_via")

    bool_cols = [c for c in train_df.select_dtypes("bool").columns if c != "risco_grave"]

    num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # num_cols.remove("risco_grave")      # target não entra
    # E remover colunas que serão tratadas nos pipelines categóricos:
    num_cols = [c for c in num_cols if "tracado_via" not in c]

    cat_pipeline = Pipeline(steps=[
        ("rare", RareLabelEncoder(tol=0.03, n_categories=1)),
        ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    tracado_pipeline = Pipeline(steps=[
        ("multilabel", MultiLabelBinarizerWrapper(separator=";"))
    ])

    num_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, cat_cols),
            ("tracado", tracado_pipeline, ["tracado_via"]),
            ("num", num_pipeline, num_cols),
            ("bool", "passthrough", bool_cols)
        ]
    )

    return pre



# ===============================================================
#                  BALANCEAMENTO DO TARGET
# ===============================================================
def undersample_dataset(df, y_col):
    true_df = df[df[y_col] == 1]
    false_df = df[df[y_col] == 0].sample(n=len(true_df), random_state=17)
    return pd.concat([true_df, false_df], axis=0)

def oversample_dataset(df, y_col):
    true_df = df[df[y_col] == 1]
    false_df = df[df[y_col] == 0]

    true_df_oversampled = resample(true_df, replace=True, n_samples=len(false_df), random_state=17)
    return pd.concat([true_df_oversampled, false_df], axis=0)

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
imbalanced_classes_method = 'oversampling' # 'smote', None, 'undersampling', 'oversampling'


datatran = load_datatran()
datatran = criar_timestamp(datatran)

# Separar OOT
cut_date = pd.to_datetime("2025-09-01")
oot_df = preprocess(datatran[datatran["timestamp"] >= cut_date])
datatran = preprocess(datatran[datatran["timestamp"] < cut_date])

# Balanceamento
y_col = "risco_grave"

if imbalanced_classes_method == 'undersampling':
    datatran = undersample_dataset(datatran, y_col)

# Criação de X e y
X = datatran.drop(columns=y_col)
y = datatran[y_col]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=17,
    stratify=y
)

if imbalanced_classes_method == 'oversampling':
    datatran_train = pd.concat([X_train, y_train], axis=1)
    datatran_train = oversample_dataset(datatran_train, y_col)

    X_train = datatran_train.drop(columns=y_col)
    y_train = datatran_train[y_col]

# Pipeline de encoding
preprocessor = criar_preprocessor(X_train)

X_train = pd.DataFrame(
    preprocessor.fit_transform(X_train),
    columns=preprocessor.get_feature_names_out()
)

X_test = pd.DataFrame(
    preprocessor.transform(X_test),
    columns=preprocessor.get_feature_names_out()
)

# SMOTE
from imblearn.over_sampling import SMOTE

if imbalanced_classes_method == 'smote':
    smote = SMOTE(random_state=17)
    X_train, y_train = smote.fit_resample(X_train, y_train)

# class weights
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)


# ---------------------------------------------------

# Treinar modelo
param_grid = {
    "n_estimators": [100, 200],
    "min_samples_leaf": [8, 10, 12],
    "max_depth": [14, 15, 16]
}

from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    n_jobs=-1,
    scoring="balanced_accuracy",
    cv=cv,
    refit=True
)

t0 = now()
rf.fit(X_train, y_train)
print("Best params:", rf.best_params_)
print(f"Treino em {now() - t0:.1f}s")

# Avaliação ROC
y_score_train = rf.best_estimator_.predict_proba(X_train)[:, 1]
auc_train = plot_roc_auc(y_train, y_score_train, label="Train")

y_score_test = rf.best_estimator_.predict_proba(X_test)[:, 1]
auc_test = plot_roc_auc(y_test, y_score_test, label="Test")

print(f"AUC train = {auc_train:.4f}")
print(f"AUC test  = {auc_test:.4f}")


# cm matrix

y_test_pred = rf.predict(X_test)

sns.heatmap(confusion_matrix(y_pred=y_test_pred, y_true=y_test), annot=True, cmap='Blues', fmt='.0f')
plt.title('Confusion Matrix Heatmap [TEST]')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

sns.heatmap(confusion_matrix(y_pred=y_test_pred, y_true=y_test, normalize='true'), annot=True, cmap='Blues', fmt='.0%')
plt.title('Confusion Matrix Heatmap [TEST]')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

feature_importances = pd.DataFrame([X_train.columns, rf.best_estimator_.feature_importances_]).T

feature_importances.columns = ['coluna', 'importancia']

feature_importances = feature_importances.sort_values(by='importancia', ascending=False)
# %%

X_oot = oot_df.drop(columns=y_col)
y_oot = oot_df[y_col]

X_oot = pd.DataFrame(
    preprocessor.transform(X_oot),
    columns=preprocessor.get_feature_names_out()
)

y_score_oot = rf.best_estimator_.predict_proba(X_oot)[:, 1]
auc_oot = plot_roc_auc(y_oot, y_score_oot, label="oot")

print(f"AUC oot  = {auc_oot:.4f}")

y_oot_pred = rf.predict(X_oot)

sns.heatmap(confusion_matrix(y_pred=y_oot_pred, y_true=y_oot), annot=True, cmap='Blues', fmt='.0f')
plt.title('Confusion Matrix Heatmap [OOT]')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# %%
