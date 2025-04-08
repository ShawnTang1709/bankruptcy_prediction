import pandas as pd
from sklearn.model_selection import train_test_split

# 讀取原始資料
input_file = "D:/Shawn/python_class/Tibame/團專/data/old/data.csv"
output_dir = "D:/Shawn/python_class/Tibame/團專/data/"

# 讀取CSV檔案
df = pd.read_csv(input_file)

# 設定目標變數和特徵
X = df.drop('Bankrupt', axis=1)  # 特徵
y = df['Bankrupt']  # 目標變數

# 使用分層抽樣方式分割資料集，test_size=0.2 表示測試集佔20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 組合回DataFrame格式
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# 儲存分割後的資料集
train_df.to_csv(output_dir + "train_data_0325.csv", index=False)
test_df.to_csv(output_dir + "test_data_0325.csv", index=False)

# 顯示分割結果
print(f"原始資料集大小: {len(df)}")
print(f"訓練集大小: {len(train_df)}")
print(f"測試集大小: {len(test_df)}")

# 確認正負類比例
print("\n類別分佈:")
print(f"原始資料 - Bankrupt=1 的比例: {y.mean():.4f}")
print(f"訓練集 - Bankrupt=1 的比例: {y_train.mean():.4f}")
print(f"測試集 - Bankrupt=1 的比例: {y_test.mean():.4f}")


