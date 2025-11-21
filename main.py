import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("zomato.csv")

# Basic structure
print(df.shape)
print(df.columns)

# General info
df.info()

# Statistical summary (numeric)
df.describe()

# Summary of categorical columns
df.describe(include='object')

# Missing values
df.isnull().sum()

# Unique values
df.nunique()


df = df.drop_duplicates()

# Clean 'rate' column (remove "NEW", "-")
df['rate'] = df['rate'].astype(str).str.replace("/5","", regex=False)
df = df[df['rate'].str.isnumeric()]
df['rate'] = df['rate'].astype(float)

# Fill missing values
df['votes'] = df['votes'].fillna(0)
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].str.replace(",","").astype(float)
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].fillna(df['approx_cost(for two people)'].median())


q_low = df['approx_cost(for two people)'].quantile(0.01)
q_hi  = df['approx_cost(for two people)'].quantile(0.99)
df = df[(df['approx_cost(for two people)'] > q_low) & (df['approx_cost(for two people)'] < q_hi)]


df['location'] = df['location'].str.strip().str.lower()
df['cuisines'] = df['cuisines'].str.lower()


df['online_order'] = df['online_order'].map({'Yes':1, 'No':0})
df['book_table'] = df['book_table'].map({'Yes':1, 'No':0})

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['votes','approx_cost(for two people)']] = scaler.fit_transform(df[['votes','approx_cost(for two people)']])
