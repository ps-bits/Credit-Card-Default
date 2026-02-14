import pandas as pd
import numpy as np

# Load the dataset
file_path = r'C:\Temp\Classification Models\data\default of credit card clients.csv'
df = pd.read_csv(file_path)

# Basic exploration
print("="*60)
print("DATASET SHAPE")
print("="*60)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n" + "="*60)
print("COLUMN NAMES")
print("="*60)
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

print("\n" + "="*60)
print("FIRST FEW ROWS")
print("="*60)
print(df.head(3))

print("\n" + "="*60)
print("DATA TYPES")
print("="*60)
print(df.dtypes)

print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)
print(df.isnull().sum())

print("\n" + "="*60)
print("TARGET VARIABLE (Last Column)")
print("="*60)
print(df.iloc[:, -1].value_counts())

print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)
print(df.describe())