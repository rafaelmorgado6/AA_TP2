# Repetir o processo agora com os ficheiros carregados

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import re

# Recarregar os dados
fake_path = "Data/Fake.csv"
true_path = "Data/True.csv"



df_fake = pd.read_csv(fake_path)
df_true = pd.read_csv(true_path)

df_fake['label'] = 0
df_true['label'] = 1

print(df_fake.columns)
print(df_true.columns)

df_fake.drop(columns=["title", "date", "subject"], inplace=True, errors='ignore')
df_true.drop(columns=["title", "date", "subject"], inplace=True, errors='ignore')

print(df_fake.columns)
print(df_true.columns)

# Limpar prefixos como "WASHINGTON (Reuters) -"
def limpar_prefixo_agencia(texto):
    return re.sub(r"^(.*?\(Reuters\)[\s\-–]*)", "", texto)

df_true['text'] = df_true['text'].apply(limpar_prefixo_agencia)

# Juntar, embaralhar e remover duplicados
df = pd.concat([df_fake, df_true], ignore_index=True)

df = df.drop_duplicates(subset='text').reset_index(drop=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Usar apenas o corpo da notícia como input
df['input'] = df['text']

# Divisão de treino/teste/validação 
X_train, X_temp, y_train, y_temp = train_test_split(
    df['input'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
# Vetorização TF-IDF
vectorizer = TfidfVectorizer(
    stop_words=None, max_features=20000, max_df=0.75, min_df=2, ngram_range=(1,2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec   = vectorizer.transform(X_val)
X_test_vec  = vectorizer.transform(X_test)

# Modelo: Regressão logística
model = LogisticRegression(
    C=100, class_weight="balanced", penalty="l2", solver="liblinear", max_iter=1000
)
model.fit(X_train_vec, y_train)

# Validação
y_val_pred = model.predict(X_val_vec)
y_val_prob = model.predict_proba(X_val_vec)[:, 1]

val_acc = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)
val_roc = roc_auc_score(y_val, y_val_prob)

print(f"Validação: Accuracy={val_acc:.4f} | F1={val_f1:.4f} | ROC AUC={val_roc:.4f}")




y_test_pred = model.predict(X_test_vec)
y_test_prob = model.predict_proba(X_test_vec)[:, 1]

test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc = roc_auc_score(y_test, y_test_prob)

print(f"Teste: Accuracy={test_acc:.4f} | F1={test_f1:.4f} | ROC AUC={test_roc:.4f}")