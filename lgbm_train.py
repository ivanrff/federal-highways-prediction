# %%
# ===============================================================
#                      IMPORTS
# ===============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, roc_curve, auc, balanced_accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MultiLabelBinarizer
from sklearn.utils import resample

from feature_engine.encoding import RareLabelEncoder

from lightgbm import LGBMClassifier

import warnings
import joblib

from time import time
from datetime import datetime

from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight

# Remover o warning do Pipeline
# warnings.filterwarnings(
#     "ignore", 
#     message="This Pipeline instance is not fitted yet.", 
#     category=FutureWarning
# )


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
    nrows_before = df.shape[0]
    df = df.dropna(how='any')
    nrows_after = df.shape[0]
    print(f"Number of rows removed: {nrows_before-nrows_after} | {100*(nrows_before-nrows_after)/nrows_before:.2f}%")
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

def rodovia_para_categ(df):
    df = df.copy()
    df['br'] = df['br'].astype(str)
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
    df = rodovia_para_categ(df)
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
    
    # --- Separação de Colunas ---
    cat_cols = train_df.select_dtypes(include="object").columns.tolist()
    if "tracado_via" in cat_cols: cat_cols.remove("tracado_via")
    
    bool_cols = [c for c in train_df.select_dtypes("bool").columns if c != "risco_grave"]
    
    num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
 
    num_cols = [c for c in num_cols if "tracado_via" not in c and c != "risco_grave"]


    # --- Pipelines Específicos ---
    

    # Pipeline Categórico (Ordinal para LGBM)
    cat_pipeline = Pipeline(steps=[
        ("rare", RareLabelEncoder(tol=0.03, n_categories=1)),
        ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    tracado_pipeline = Pipeline(steps=[
        ("multilabel", MultiLabelBinarizerWrapper(separator=";"))
    ])

    # num_pipeline = Pipeline(steps=[
    #     ("scaler", StandardScaler())
    # ])

    # --- Juntar Tudo ---
    pre = ColumnTransformer(
        transformers=[

            # ("geo", geo_pipeline, ["latitude", "longitude"]),
            
            ("cat", cat_pipeline, cat_cols),
            ("tracado", tracado_pipeline, ["tracado_via"]),
            ("num", "passthrough", num_cols),
            ("bool", "passthrough", bool_cols)
        ],
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # --- Atualizar lista de Categóricas para o LGBM ---
    # Originais + a nova 'geo_cluster' que criamos
    features_categoricas = cat_cols

    return pre, features_categoricas


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

imbalanced_classes_method = 'undersampling' # 'smote', None, 'undersampling', 'oversampling'


datatran = load_datatran()
datatran = criar_timestamp(datatran)

# Separar OOT
cut_date = pd.to_datetime("2025-09-01")
no_oot_df = preprocess(datatran[datatran["timestamp"] < cut_date])

# Criação de X e y
y_col = "risco_grave"
X = no_oot_df.drop(columns=y_col)
y = no_oot_df[y_col]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=17,
    stratify=y
)

# Checando desbalanceamento
print(f"Rate of 1 (True) class: {100*y_train.sum()/len(y_train):.2f}%")


if imbalanced_classes_method == 'undersampling':
    datatran_train = pd.concat([X_train, y_train], axis=1)
    datatran_train = undersample_dataset(datatran_train, y_col)

    X_train = datatran_train.drop(columns=y_col)
    y_train = datatran_train[y_col]
elif imbalanced_classes_method == 'oversampling':
    datatran_train = pd.concat([X_train, y_train], axis=1)
    datatran_train = oversample_dataset(datatran_train, y_col)

    X_train = datatran_train.drop(columns=y_col)
    y_train = datatran_train[y_col]

# Pipeline de encoding
preprocessor, features_categoricas = criar_preprocessor(X_train)

# Class Weights (só é utilizado se o modelo sendo testado não tiver o argumento de class_weights por natureza)
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# scale_pos_weight (só é aplicado em alguns modelos)
n_neg = (y_train[y_train == 0]).shape[0]
n_pos = (y_train[y_train == 1]).shape[0]
print(f"Classe [0]: {n_neg}; Classe [1]: {n_pos}" )

scale = n_neg / n_pos

# ---------------------------------------------------

# Treinar modelo

model_pca_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)), # este passo é desativado logo abaixo caso imbalanced_classes_method != 'smote'
    ('lgbm', LGBMClassifier(random_state=42, scale_pos_weight=scale)) # , class_weight='balanced'))
])

# Condicional para ativar o passo de SMOTE da pipeline
if imbalanced_classes_method != 'smote':
    model_pca_pipeline.set_params(smote='passthrough')
else:
    print("Aplicando SMOTE...")

param_grid_lgbm = {
    # Estrutura da árvore
    "lgbm__num_leaves": [15, 25],
    "lgbm__max_depth": [15, 20],
    
    # Aprendizado
    "lgbm__learning_rate": [0.01, 0.05],
    "lgbm__n_estimators": [300, 400]
}

param_grid_lgbm_robust = {
    # Estrutura da árvore
    "lgbm__num_leaves": [31, 64, 128], 
    "lgbm__max_depth": [-1],
    "lgbm__min_child_samples": [20, 50],

    # Aprendizado
    "lgbm__learning_rate": [0.01, 0.05],
    "lgbm__n_estimators": [500, 1000]

}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

lgbm_gs = GridSearchCV(
    estimator=model_pca_pipeline,
    param_grid=param_grid_lgbm_robust,
    n_jobs=-1,
    scoring="balanced_accuracy",
    cv=cv,
    refit=True
)


t0 = time()
print(f"Treino iniciado {datetime.now()}")
lgbm_gs.fit(X_train, y_train, lgbm__categorical_feature=features_categoricas)
print("Best params:", lgbm_gs.best_params_)
print(f"Treino em {time() - t0:.1f}s")

def evaluate(model, X_to_predict, y_true, data_slice_name):

    # Avaliação ROC
    y_probas = lgbm_gs.best_estimator_.predict_proba(X_to_predict)[:, 1]
    auc = plot_roc_auc(y_true, y_probas, label=data_slice_name)

    print(f"AUC {data_slice_name} = {auc:.4f}")

    y_pred = model.predict(X_to_predict)

    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"Bal ACC {data_slice_name} = {bal_acc}")

    # cm matrix
    sns.heatmap(confusion_matrix(y_pred=y_pred, y_true=y_true), annot=True, cmap='Blues', fmt='.0f')
    plt.title(f'Confusion Matrix Heatmap [{data_slice_name}]')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    sns.heatmap(confusion_matrix(y_pred=y_pred, y_true=y_true, normalize='true'), annot=True, cmap='Blues', fmt='.0%')
    plt.title(f'Confusion Matrix Heatmap [{data_slice_name}]')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

evaluate(lgbm_gs, X_train, y_train, 'TRAIN')
evaluate(lgbm_gs, X_test, y_test, 'TEST')

# %%

# ===============================================================
#              AVALIAÇÃO DE IMPORTÂNCIA DE FEATURES
# ===============================================================

booster = lgbm_gs.best_estimator_.named_steps['lgbm'].booster_
importancias = booster.feature_importance(importance_type='gain')

# Recuperar nomes das features
feature_names = lgbm_gs.best_estimator_.named_steps['preprocessor'].get_feature_names_out()

# Limpeza dos nomes
feature_names = [f.split('__')[-1] for f in feature_names]

df_imp = pd.DataFrame({
    'Feature': feature_names,
    'Gain': importancias
}).sort_values(by='Gain', ascending=False)

print(df_imp.head(20))

plt.figure(figsize=(10, 8))
sns.barplot(x='Gain', y='Feature', data=df_imp.head(20))
plt.title('O que realmente importa para o modelo?')
plt.show()

# %%

# ===============================================================
#                   AVALIAÇÃO NO OUT-OF-TIME
# ===============================================================
print("Preprocessing OOT dataset")
oot_df = preprocess(datatran[datatran["timestamp"] >= cut_date])

X_oot = oot_df.drop(columns=y_col)
y_oot = oot_df[y_col]

evaluate(lgbm_gs, X_oot, y_oot, 'OOT')

# %%

# ===============================================================
#                       EXPORTAR O .pkl
# ===============================================================

# Salvar o melhor modelo encontrado pelo GridSearch
joblib.dump(lgbm_gs.best_estimator_, 'modelo_acidentes_geocluster.pkl')

print("Modelo salvo como 'modelo_acidentes.pkl'")

# %%
