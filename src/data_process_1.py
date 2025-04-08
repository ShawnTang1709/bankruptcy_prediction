import pandas as pd
import os
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
from sklearn.model_selection import train_test_split

# 1. 讀取資料
input_file_1 = "D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/data/35/test_data_0324.csv"
input_file_2 = "D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/data/35/train_data_0324.csv"
output_dir = "D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/data/35/done/ori"

# 2. 載入CSV檔案
df_test = pd.read_csv(input_file_1)
df_train = pd.read_csv(input_file_2)

# 需要保留的欄位名稱（原始格式）
# columns_to_keep_original = [
#     'Operating Gross Margin', 
#     'Tax rate (A)', 
#     'Net Value Per Share (A)', 
#     'Persistent EPS in the Last Four Seasons', 
#     'Cash Flow Per Share', 
#     'Revenue Per Share (Yuan)', 
#     'After-tax Net Profit Growth Rate', 
#     'Quick Ratio', 
#     'Total debt/Total net worth', 
#     'Debt ratio ', 
#     'Contingent liabilities/Net worth', 
#     'Inventory and accounts receivable/Net value', 
#     'Total Asset Turnover', 
#     'Operating profit per person',
#     'Allocation rate per person', 
#     'Working Capital to Total Assets', 
#     'Quick Assets/Total Assets', 
#     'Current Assets/Total Assets', 
#     'Cash/Total Assets', 
#     'Cash/Current Liability',
#     'Operating Funds to Liability', 
#     'Working Capital/Equity', 
#     'Long-term Liability to Current Assets', 
#     'Total expense/Assets', 
#     'Fixed Assets to Assets', 
#     'Equity to Long-term Liability', 
#     'Cash Flow to Total Assets', 
#     'Cash Flow to Liability', 
#     'CFO to Assets', 
#     'Cash Flow to Equity', 
#     'Current Liability to Current Assets', 
#     'Liability-Assets Flag', 
#     'Net Income to Total Assets', 
#     "Net Income to Stockholder's Equity", 
#     'Equity to Liability']

# 創建原始欄位名稱到清理後欄位名稱的映射
# 清理：移除空格、括號、特殊符號等
def clean_column_name(col_name):
    return col_name.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace('(', '').replace(')', '').replace('%', '').replace('.', '')

# 函數：處理數據集並選取需要的欄位
def process_dataset(df, name):
    print(f"\n處理{name}集...")
    # 如果資料中有標籤欄位，保留它
    target_column = None
    for label_col in ['target', 'Bankrupt', 'Y', 'label']:
        if label_col in df.columns:
            print(f"已找到標籤欄位: {label_col}")
            target_column = label_col
            break
    
    return df, target_column  # 返回完整的數據框和目標欄位

# 處理訓練集和測試集
filtered_train, train_target_column = process_dataset(df_train, "訓練")
filtered_test, test_target_column = process_dataset(df_test, "測試")

# 在進行抽樣處理之前，先分割訓練集和驗證集
if filtered_train is not None and filtered_test is not None:
    # 儲存原始測試集
    output_test_file = os.path.join(output_dir, "filtered_test_data.csv")
    filtered_test.to_csv(output_test_file, index=False)
    print(f"處理後的測試集已儲存至: {output_test_file}")
    print(f"測試集維度: {filtered_test.shape}")
    
    # 只對訓練集進行抽樣
    if train_target_column is not None:
        print(f"\n分割訓練集和驗證集...")
        # 分割訓練集和驗證集（80% 訓練，20% 驗證）
        train_data, val_data = train_test_split(filtered_train, test_size=0.2, random_state=42, stratify=filtered_train[train_target_column])
        
        print(f"訓練集維度: {train_data.shape}")
        print(f"驗證集維度: {val_data.shape}")
        
        # 儲存驗證集
        output_val_file = os.path.join(output_dir, "filtered_val_data.csv")
        val_data.to_csv(output_val_file, index=False)
        print(f"驗證集已儲存至: {output_val_file}")
        
        # 查看類別分佈
        print("\n訓練集目標欄位的分佈狀況：")
        train_value_counts = train_data[train_target_column].value_counts()
        print(train_value_counts)
        print(f"不平衡比例：{train_value_counts.max() / train_value_counts.min():.2f}:1")

        # 分離特徵和目標變數（使用train_data而不是filtered_train）
        X_train = train_data.drop(train_target_column, axis=1)
        y_train = train_data[train_target_column]
        
        # 儲存原始訓練集
        output_train_file = os.path.join(output_dir, "filtered_train_data_original.csv")
        train_data.to_csv(output_train_file, index=False)
        
        # 實現不同的抽樣方法
        sampling_methods = {
            'random_undersampling': RandomUnderSampler(random_state=42),
            'smote_tomek': SMOTETomek(sampling_strategy=0.2,random_state=42),
            'smote': SMOTE(random_state=42, sampling_strategy=0.2),
            'adasyn': ADASYN(random_state=42),
            'smote_enn': SMOTEENN(sampling_strategy=0.2, random_state=42)
        }
        
        # 儲存每種抽樣方法的結果
        for method_name, sampler in sampling_methods.items():
            print(f"\n應用 {method_name} 抽樣方法...")
            try:
                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                
                # 合併特徵和目標變數
                resampled_df = pd.DataFrame(X_resampled, columns=X_train.columns)
                resampled_df[train_target_column] = y_resampled
                
                # 計算新的類別分佈
                new_counts = Counter(y_resampled)
                print(f"{method_name} 抽樣後的分佈: {new_counts}")
                
                # 儲存處理後的資料
                output_file = os.path.join(output_dir, f"filtered_train_data_{method_name}.csv")
                resampled_df.to_csv(output_file, index=False)
                print(f"{method_name} 處理後的訓練資料已儲存至: {output_file}")
                print(f"資料維度: {resampled_df.shape}")
            except Exception as e:
                print(f"應用 {method_name} 時發生錯誤: {str(e)}")
    else:
        print("警告：找不到訓練集的目標欄位！")
else:
    if filtered_train is None:
        print("無法處理訓練集")
    if filtered_test is None:
        print("無法處理測試集")

print("\n資料處理完成!")
