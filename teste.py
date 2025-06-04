import pandas as pd
import matplotlib.pyplot as plt

# Carregamento dos dados
true_df = pd.read_csv('Data/True.csv')
fake_df = pd.read_csv('Data/Fake.csv')

# Adição de rótulo
true_df['label'] = 'True'
fake_df['label'] = 'Fake'

# Combinação
df = pd.concat([true_df, fake_df], ignore_index=True)

# Conversão da data
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# 🔁 Agrupamento manual dos temas
def map_subject(subj):
    if subj in ['politicsNews', 'politics', 'Government News', 'left-news']:
        return 'Politics'
    elif subj in ['worldnews', 'Middle-east']:
        return 'World'
    elif subj in ['News', 'US_News']:
        return 'General'
    else:
        return 'Other'

df['subject_group'] = df['subject'].apply(map_subject)

# 1. Gráfico circular: distribuição dos temas agrupados
subject_group_counts = df['subject_group'].value_counts()
fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.pie(subject_group_counts, labels=subject_group_counts.index, autopct='%1.1f%%', startangle=140)
ax1.set_title("Distribuição dos Temas Agrupados")
plt.tight_layout()
plt.show()

# 2. Histograma: número de notícias por mês
monthly_counts = df['date'].dt.to_period('M').value_counts().sort_index()
fig2, ax2 = plt.subplots(figsize=(12, 6))
monthly_counts.plot(kind='bar', ax=ax2)
ax2.set_title("Número de Notícias por Mês")
ax2.set_xlabel("Data")
ax2.set_ylabel("Número de Notícias")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Gráfico de barras: True vs Fake por tema agrupado
group_label_counts = df.groupby(['subject_group', 'label']).size().unstack().fillna(0)
fig3, ax3 = plt.subplots(figsize=(10, 6))
group_label_counts.plot(kind='bar', stacked=True, ax=ax3)
ax3.set_title("Notícias Verdadeiras vs Falsas por Tema Agrupado")
ax3.set_xlabel("Tema Agrupado")
ax3.set_ylabel("Número de Notícias")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
