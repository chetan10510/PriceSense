import pandas as pd


# Step 1: Load the dataset

# Use raw string or forward slashes to avoid escape sequence issues
df = pd.read_csv(r"data\processed\amazon_laptops_prices_filtered.csv")


# Step 2: Basic inspection

print("[INFO] Dataset shape:", df.shape)
print("[INFO] Columns:", df.columns.tolist())
print(df.head())


# Step 3: Feature Engineering


# 3a. Extract brand from title (first word)
df['brand'] = df['title'].str.split().str[0]

# 3b. Extract numeric RAM (if present in title, e.g., "16GB")
import re
def extract_ram(text):
    match = re.search(r'(\d+)\s*GB', text, re.IGNORECASE)
    return int(match.group(1)) if match else None

df['ram_gb'] = df['title'].apply(extract_ram)

# 3c. Extract numeric storage (SSD/HDD)
def extract_storage(text):
    match = re.search(r'(\d+)\s*(GB|TB)', text, re.IGNORECASE)
    if match:
        size = int(match.group(1))
        unit = match.group(2).upper()
        if unit == 'TB':
            size *= 1024  # convert TB to GB
        return size
    return None

df['storage_gb'] = df['title'].apply(extract_storage)

# 3d. Clean price column (remove currency symbols, commas)
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)


# Step 4: Optional – Drop rows with missing critical features

df = df.dropna(subset=['price', 'ram_gb', 'storage_gb'])


# Step 5: Save processed dataset

df.to_csv(r"data/processed/amazon_laptops_features.csv", index=False)
print("[INFO] Feature-engineered dataset saved → data/processed/amazon_laptops_features.csv")
