import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO)

# Load scraped dataset
df = pd.read_csv('data/processed/amazon_laptops_features.csv')

logging.info(f"Original dataset shape: {df.shape}")

# Fill missing ratings
df['rating'] = df['rating'].fillna(df['rating'].median())

# Function to extract RAM in GB
def extract_ram(title):
    match = re.search(r'(\d+)\s*GB\s*RAM', title, re.IGNORECASE)
    return int(match.group(1)) if match else None

# Function to extract storage in GB
def extract_storage(title):
    match = re.search(r'(\d+)\s*(GB|TB)\s*(SSD|HDD)?', title, re.IGNORECASE)
    if match:
        size = int(match.group(1))
        unit = match.group(2)
        if unit.upper() == 'TB':
            size *= 1024
        return size
    return None

# Function to extract CPU brand and generation
def extract_cpu(title):
    cpu_patterns = ['Intel i3', 'Intel i5', 'Intel i7', 'Intel i9', 
                    'AMD Ryzen 3', 'AMD Ryzen 5', 'AMD Ryzen 7', 'AMD Ryzen 9', 'AMD Athlon']
    for cpu in cpu_patterns:
        if cpu.lower() in title.lower():
            return cpu
    return 'Other'

# Function to extract screen size
def extract_screen(title):
    match = re.search(r'(\d{2}\.?\d?)\s*-?inch', title, re.IGNORECASE)
    return float(match.group(1)) if match else None

# Function to check if discrete GPU mentioned
def has_gpu(title):
    keywords = ['NVIDIA', 'GeForce', 'RTX', 'GTX', 'MX']
    return int(any(k.lower() in title.lower() for k in keywords))

# Extract features
df['brand'] = df['title'].apply(lambda x: x.split()[0])
df['ram_gb'] = df['title'].apply(extract_ram)
df['storage_gb'] = df['title'].apply(extract_storage)
df['cpu_type'] = df['title'].apply(extract_cpu)
df['screen_inch'] = df['title'].apply(extract_screen)
df['gpu'] = df['title'].apply(has_gpu)

# Drop original title column (optional)
df = df.drop(columns=['title', 'link'])

# Fill missing numeric values with median
numeric_cols = ['ram_gb', 'storage_gb', 'screen_inch']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

logging.info(f"Enhanced dataset shape: {df.shape}")
logging.info(f"Columns: {df.columns.tolist()}")

# Save enhanced features
df.to_csv('data/processed/amazon_laptops_features_enhanced.csv', index=False)
logging.info("Enhanced feature dataset saved → data/processed/amazon_laptops_features_enhanced.csv")
