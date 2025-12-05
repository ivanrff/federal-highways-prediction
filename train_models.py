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
        "id", "timestamp", "veiculos", "pessoas"
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
imbalanced_classes_method = 'undersampling' # 'smote', None, 'undersampling', 'oversampling'


datatran = load_datatran()
datatran = criar_timestamp(datatran)

# Separar OOT
cut_date = pd.to_datetime("2025-09-01")
oot_df = preprocess(datatran[datatran["timestamp"] >= cut_date])
datatran = preprocess(datatran[datatran["timestamp"] < cut_date])

# Balanceamento
y_col = "risco_grave"

# Criação de X e y
X = datatran.drop(columns=y_col)
y = datatran[y_col]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=17,
    stratify=y
)

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

# Treinar modelo
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score
from imblearn.metrics import specificity_score

modelos = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    # "SVC": SVC(kernel='rbf'),
    "KNN": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier()
}

if not imbalanced_classes_method:
    n_neg = (datatran[datatran['risco_grave'] == 0]).shape[0]
    print(n_neg)
    n_pos = (datatran[datatran['risco_grave'] == 1]).shape[0]
    print(n_pos)

    scale = n_neg / n_pos

    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced'),
        # "SVC": SVC(kernel='rbf', class_weight='balanced'),
        "Random Forest": RandomForestClassifier(class_weight='balanced'),
        "XGBoost": XGBClassifier(scale_pos_weight=(scale)),
        "LightGBM": LGBMClassifier(class_weight='balanced')
    }

# modelos = {
#     "Logistic Regression": LogisticRegression(max_iter=2000),
#     "Random Forest": RandomForestClassifier(),
#     "XGBoost": XGBClassifier(),
#     "LightGBM": LGBMClassifier()
# }

resultados = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

for nome, modelo in tqdm(modelos.items()):

    resultados_cv = cross_validate(
                                modelo,
                                X_train, y_train,
                                cv=5,
                                return_train_score=True,
                                scoring=['balanced_accuracy', 'roc_auc']
                            )
    if not imbalanced_classes_method:
        try:
            modelo.fit(X_train, y_train, sample_weight=sample_weights)
        except:
            print(f"{nome} falhou o sample_weight")
            modelo.fit(X_train, y_train)
    else:
        modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    bal_acc_train = balanced_accuracy_score(y_true=y_train, y_pred=y_pred_train)
    bal_acc_test = balanced_accuracy_score(y_true=y_test, y_pred=y_pred_test)

    recall_test = recall_score(y_true=y_test, y_pred=y_pred_test)
    specificity_test = specificity_score(y_true=y_test, y_pred=y_pred_test)

    # Média dos resultados por modelo
    resultados[nome] = {
        "cv_train_bal_acc_mean": resultados_cv["train_balanced_accuracy"].mean(),
        # "cv_train_bal_acc_std":  resultados_cv["train_balanced_accuracy"].std(),
        "cv_test_bal_acc_mean":  resultados_cv["test_balanced_accuracy"].mean(),
        # "cv_test_bal_acc_std":   resultados_cv["test_balanced_accuracy"].std(),

        "train_bal_acc": bal_acc_train,
        "test_bal_acc": bal_acc_test,
        "test_recall": recall_test,
        "test_specificity": specificity_test,

        "cv_train_roc_auc_mean": resultados_cv["train_roc_auc"].mean(),
        # "cv_train_roc_auc_std":  resultados_cv["train_roc_auc"].std(),
        "cv_test_roc_auc_mean":  resultados_cv["test_roc_auc"].mean(),
        # "cv_test_roc_auc_std":   resultados_cv["test_roc_auc"].std()
    }

df_resultados = pd.DataFrame(resultados).T.sort_values(by='test_bal_acc', ascending=False)

os.makedirs('results', exist_ok=True)

with open(file=f'results/models_results_{imbalanced_classes_method}.md', mode='w') as file:
    file.write(df_resultados.to_markdown())
# %%

# coletar os resultados com smote, com undersampling e sem nada e salvar no readme