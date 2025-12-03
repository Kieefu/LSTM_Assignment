import pandas as pd

# Read Excel file
print("Reading Excel file...")
df = pd.read_excel('cleaned_weather.csv.xlsx')

# Display basic info
print(f"\nDataset shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n" + "="*60)
print("Column Names and Types:")
print("="*60)
print(df.dtypes)

print("\n" + "="*60)
print("First 10 rows:")
print("="*60)
print(df.head(10))

print("\n" + "="*60)
print("Statistical Summary:")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("Missing Values:")
print("="*60)
print(df.isnull().sum())

# Save as CSV
print("\n" + "="*60)
print("Saving as CSV...")
df.to_csv('cleaned_weather.csv', index=False)
print("Saved to: cleaned_weather.csv")
print("="*60)
