import pandas as pd
import matplotlib.pyplot as plt

# Carregamento
fake_df = pd.read_csv('Data/Fake.csv')
true_df = pd.read_csv('Data/True.csv')

# Contar temas
subject_counts_fake = fake_df['subject'].value_counts()
subject_counts_true = true_df['subject'].value_counts()

# Adicionar label
true_df['label'] = 'True'
fake_df['label'] = 'Fake'

# ✅ Converter a coluna 'date' ANTES de juntar
true_df['date'] = pd.to_datetime(true_df['date'], errors='coerce', dayfirst=True)
fake_df['date'] = pd.to_datetime(fake_df['date'], errors='coerce', dayfirst=True)

# Verificação real agora
print("True com data:", true_df['date'].notna().sum())
print("Fake com data:", fake_df['date'].notna().sum())

# Combinar
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.dropna(subset=['date'])

# Agrupar por mês
df['month'] = df['date'].dt.to_period('M')
monthly_counts_by_label = df.groupby(['month', 'label']).size().unstack(fill_value=0)

# Gráfico circular (Fake)
plt.figure(figsize=(8, 8))
plt.pie(subject_counts_fake, labels=subject_counts_fake.index, autopct='%1.1f%%', startangle=140)
plt.title("Fake news - Subject Distribution")
plt.tight_layout()
plt.show()

# Gráfico circular (True)
plt.figure(figsize=(8, 8))
plt.pie(subject_counts_true, labels=subject_counts_true.index, autopct='%1.1f%%', startangle=140)
plt.title("True news - Subject Distribution")
plt.tight_layout()
plt.show()

# Gráfico de barras por mês
monthly_counts_by_label.plot(kind='bar', figsize=(14, 6))
plt.title('Number of Fake and True News Articles by Month')
plt.xlabel('Month')
plt.ylabel('Number of News Articles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()