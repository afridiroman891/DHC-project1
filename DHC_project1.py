#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Load and Inspect the Dataset
# ==========================================


df = pd.read_csv("D:\\iris_dataset (1).csv")


print("--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df.shape)


print("\n--- Column Names ---")
print(df.columns.tolist())


print("\n--- First 5 Rows (.head()) ---")
print(df.head())

print("\n--- Dataset Info Summary ---")
df.info()

print("\n--- Numerical Summary Statistics (.describe()) ---")
print(df.describe(include='all'))


# ==========================================
# 2. Visualize the Dataset
# ==========================================

sns.set_theme(style="whitegrid")
features = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='petal_length_cm', y='petal_width_cm', hue='species', palette='Set1', s=70)
plt.title('Scatter Plot: Petal Length vs Petal Width', fontsize=14, pad=15)
plt.xlabel('Petal Length (cm)', fontsize=12)
plt.ylabel('Petal Width (cm)', fontsize=12)
plt.legend(title='Species')
plt.tight_layout()
plt.savefig('iris_scatter_plot.png') 
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Histograms: Feature Value Distributions by Species', fontsize=16, y=0.98)

for ax, feature in zip(axes.flatten(), features):
    sns.histplot(data=df, x=feature, hue='species', kde=True, ax=ax, palette='Set1', multiple='stack')
    ax.set_title(f'Distribution of {feature.replace("_", " ").title()}', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('iris_histograms.png')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Box Plots: Feature Dimensions and Outlier Identification', fontsize=16, y=0.98)

for ax, feature in zip(axes.flatten(), features):
    sns.boxplot(data=df, x='species', y=feature, ax=ax, palette='Set1')
    ax.set_title(f'{feature.replace("_", " ").title()} by Species', fontsize=12)
    ax.set_xlabel('Species', fontsize=10)
    ax.set_ylabel('Length/Width (cm)', fontsize=10)

plt.tight_layout()
plt.savefig('iris_box_plots.png')
plt.show()

pair_plot = sns.pairplot(df, hue='species', palette='Set1', diag_kind='kde')
pair_plot.fig.suptitle('Pairplot Matrix of Iris Features', y=1.02, fontsize=16)
pair_plot.savefig('iris_pairplot.png')
plt.show()


# In[ ]:




