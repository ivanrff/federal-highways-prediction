# Predicting Fatally or Gravely Wounded Victims

This project goal was to build a machine learning model to predict if a road accident had fatally or gravely wounded victims, so we are dealing with a classification problem that should output 1 if the example had any fatally or gravely wounded victims and 0 if it hadn't. A group of models were trained on train_models.py and after that a LightGBM model was trained on `lgbm_train.py`.

## Data Understanding

The data was taken from the PRF (Polícia Rodoviária Federal, Federal Highway Police) website, for the 2023, 2024 and 2025 years.

### Columns

- `id`
- `data_inversa`: YYYY-MM-DD
- `dia_semana`: week day in portuguese, `string`. E.g. wednesday `quarta-feira`
- `horario`: HH:MM:SS
- `uf`: state, `string`. E.g. Rio de Janeiro `RJ`, São Paulo `SP`, etc.
- `br`: Highway identification number (E.g. BR101 would be `101` here)
- `km`: Highway kilometer number (E.g. BR101's first kilometer would be `1` here)
- `municipio`: city
- `causa_acidente`: description of what caused the accident (E.g. driver snoozed)
- `tipo_acidente`: type of accident (E.g. frontal collision `colisão frontal`, rear collision `colisão traseira`)
- `classificacao_acidente`: if it had fatal victims, injured ones, etc.
- `fase_dia`: like day `dia`, night `noite`...
- `sentido_via`: direction of the lane, forwards or backwards, basically
- `condicao_metereologica`: weather conditions
- `tipo_pista`: type of lane, single lane, two lanes...
- `tracado_via`: straight road, curve...
- `uso_solo`: yes `Sim` if it's an unpaved road
- `pessoas`: number of people involved in the accident
- `mortos`: number of deceased
- `feridos_leves`: number of lightly injured people
- `feridos_graves`: number of gravely injured people
- `ilesos`: number of unharmed people
- `ignorados`: number of people with unknown state after the incident
- `feridos`: number of people that got hurt
- `veiculos`: number of vehicles involved
- `latitude`: self explanatory
- `longitude`: self explanatory
- `regional`, `delegacia`, `uop`: police division responsible for the region the accident occurred.

## Data Preparation

### Target column

For this scenario, our target column, `risco_grave`, consists of a boolean, which takes value `1` if the accident had any fatally OR gravely wounded victims, else it takes `0`.

The future idea would be to implement an app that takes a starting point and time and a destination and slice the trip into smaller distances, for example each kilometer of a highway. For each distance, the app would assume an accident and check if there was any gravely or fatally wounded. If yes, it would warn the user to be extremely careful. This would be useful for GPS tracking apps, for example.

We have a case of data imbalance for this target column. Roughly 28% of the rows are True for the target column. It was addressed during training.

### Features

1. After investigating, no duplicated data was found and only a few lines had empty values (<0.1% of total rows).
2. `timestamp` column was created from `horario` e `data_inversa`.

At this point, with `timestamp` the dataset was sliced, removing the last month as an out of time sample, for post model evaluating.

3. Month `mes` column was created, which takes values 1-12.
4. `hora_sin` and `hora_cos` were created to house the cyclical nature of the hour.
    - Otherwise having the hour as a number of 0 to 23 would make the model learn that 23 is the farthest hour from 0, whereas the reality is that 23h and 0h are as close as 1h and 2h, for example.
5. Some columns were removed:
    - `mortos`, `feridos_leves`, `feridos_graves`, `ilesos`, `ignorados`, `feridos`, `classificacao_acidente`, `tipo_acidente`, `causa_acidente`, `pessoas`, `veiculos`: all these columns give information about the situation after the accident, which wouldn't be present in a GPS tracking scenario. Also some of these columns would result in data leakage.
    - `id` isn't usable for the model, and we already have time information from other columns.
    - `municipio`, `delegacia`, `regional`: these give location information which we already have from `uop`.
6. For `km`, `latitude` and `longitude`, it was required to replace the commas with dots so pandas recognizes them as numbers.
7. `dia_semana` was changed from strings to numbers 0-6 and then it received the same cyclical sin-cos treatment.
8. `uso_solo` mapped as `{'Sim': 1, 'Não': 0}`

Also, `br` column although it's a number, should be considered a category. For example, the `101` of BR101 is just its name. The number doesn't hold any numerical relevance.

All these transformations were wrapped into a `preprocess()` function.

#### Categorical columns

* `tracado_via` is a unique column in the sense it holds 1 or more informations about the road. For example, a sample can be 
`'Túnel;Interseção de Vias;Reta'` (_tunnel, crossroads, straight road_), or `'Reta'` (_straight road_) or `'Interseção de Vias;Ponte'` (_crossroads, bridge_). So, for that a `MultiLabelBinarizer()` technique was applied.
* A `RareLabelEncoder()` was also applied so that values that appear very rarely (\<3% occurrences) on categorical columns are swapped for an `other` value.
* `OneHotEncoder()` was then applied to the categorical columns except `tracado_via`.
    * As a feature of LightGBM, we can `OrdinalEncoder()` the categorical columns instead and pass the list of columns as a parameter to the model so it understands they are not numerical columns.

All these encoders were applied through pipelines, so they could be reproduced in the OOT dataset.

## Modelling

The dataset was split again into train and test. For the first modelling test, a group of models was chosen (`train_models.py`):

```
modelos = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier()
}
```

For these models, the imbalaced dataset was handled via four techniques.
- Undersampling (removing rows from the predominant class until the data gets close to 50/50 rate)
- SMOTE
- Oversampling (replicating rows)
- None (using the `class_weights` parameter from some models)

All these techniques yielded similar results, with LightGBM reaching the highest balanced accuracy in the test dataset (0.59~0.61) with undersampling and oversampling applied.

The following results were collected at `lgbm_train.py`.

### Train dataset

Balanced Accuracy: 0.65

![ROC Curve for train dataset](/results/lgbm_train/roc_train.png)
![Confusion Matrix for train dataset](/results/lgbm_train/cm_train.png)

### Test dataset

Balanced Accuracy: 0.59

![ROC Curve for test dataset](/results/lgbm_train/roc_test.png)
![Confusion Matrix for test dataset](/results/lgbm_train/cm_test.png)

### OOT dataset

Balanced Accuracy: 0.58

![ROC Curve for oot dataset](/results/lgbm_train/roc_oot.png)
![Confusion Matrix for oot dataset](/results/lgbm_train/cm_oot.png)