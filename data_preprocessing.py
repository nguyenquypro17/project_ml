import pandas as pd
import os

df = pd.read_csv('/Users/ducquynguyen/Downloads/data for bioinformatics/data_phase1.csv', sep=None, engine='python') 

# Đặt lại tên cột
df.columns = ['text', 'label']

# --- BƯỚC MỚI: Xóa dấu chấm thừa ở cuối câu ---
df['text'] = df['text'].str.rstrip('.') 

# Tách file
train = df[df['label'] == 'training']
valid = df[df['label'] == 'validation']
test  = df[df['label'] == 'testing']

# Lưu file
os.makedirs('dataset', exist_ok=True)
train.to_csv('dataset/train.csv', index=False)
valid.to_csv('dataset/valid.csv', index=False)
test.to_csv('dataset/test.csv', index=False)

print("Xong! Đã xóa dấu chấm cuối câu.")