import pandas as pd
import os
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter

# 1. 讀取資料
input_file_1 = "D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/data_process/test_data_0314_rn.csv"
input_file_2 = "D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/data_process/train_data_0314_rn.csv"
output_dir = "D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/data_process/68"

# 2. 載入CSV檔案
df_test = pd.read_csv(input_file_1)
df_train = pd.read_csv(input_file_2)

# 需要保留的欄位名稱（原始格式）
columns_to_keep_original = [
    'Operating_Funds_to_Liability',
    'Current_Liability_to_Current_Assets',
    'Net_Value_Per_Share_A',
    'Cash_Flow_to_Liability',
    'Current_Assets_Total_Assets',
    'Operating_Gross_Margin',
    'Quick_Assets_Total_Assets',
    'Total_Asset_Turnover',
    'Net_Worth_Turnover_Rate_times',
    'After_tax_net_Interest_Rate',
    'Degree_of_Financial_Leverage_DFL',
    'Tax_rate_A',
    'Continuous_Net_Profit_Growth_Rate',
    'Cash_Current_Liability',
    'Interest_Coverage_Ratio_Interest_EBIT',
    'Contingent_liabilities_Net_worth',
    'Cash_Reinvestment',
    'Cash_Flow_Per_Share',
    'Cash_flow_rate',
    'Realized_Sales_Gross_Margin',
    'ROA_A_before_interest_and_after_tax',
    'Operating_Profit_Growth_Rate',
    'Revenue_Per_Share_Yuan',
    'Operating_profit_per_person',
    'Persistent_EPS_in_the_Last_Four_Seasons',
    'Net_Income_to_Stockholder_Equity',
    'Total_debt_Total_net_worth',
    'Long_term_fund_suitability_ratio_A',
    'Working_Capital_Equity',
    'Current_Liabilities_Liability',
    'Inventory_and_accounts_receivable_Net_value',
    'Quick_Assets_Current_Liability',
    'Net_Income_to_Total_Assets',
    'Current_Liability_to_Assets',
    'Net_Value_Growth_Rate',
    'Net_Value_Per_Share_B',
    'Quick_Ratio',
    'Total_income_Total_expense',
    'No_credit_Interval',
    'Realized_Sales_Gross_Profit_Growth_Rate',
    'Total_Asset_Return_Growth_Rate_Ratio',
    'Cash_Flow_to_Equity',
    'Inventory_Current_Liability',
    'Accounts_Receivable_Turnover',
    'ROA_B_before_interest_and_depreciation_after_tax',
    'Long_term_Liability_to_Current_Assets',
    'Cash_Total_Assets',
    'Fixed_Assets_to_Assets',
    'Average_Collection_Days',
    'Interest_Expense_Ratio',
    'Retained_Earnings_to_Total_Assets',
    'Non_industry_income_and_expenditure_revenue',
    'Operating_Profit_Rate',
    'Inventory_Working_Capital',
    'Equity_to_Long_term_Liability',
    'Cash_Flow_to_Sales',
    'ROA_C_before_interest_and_depreciation_before_interest',
    'Borrowing_dependency',
    'Total_assets_to_GNP_price',
    'Revenue_per_person',
    'Total_expense_Assets',
    'Allocation_rate_per_person',
    'Interest_bearing_debt_interest_rate'
]

# 創建原始欄位名稱到清理後欄位名稱的映射
# 清理：移除空格、括號、特殊符號等
def clean_column_name(col_name):
    return col_name.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace('(', '').replace(')', '').replace('%', '').replace('.', '')

# 函數：處理數據集並選取需要的欄位
def process_dataset(df, name):
    print(f"\n處理{name}集...")
    # 建立映射字典
    column_mapping = {clean_column_name(col): col for col in df.columns}
    needed_columns_clean = [clean_column_name(col) for col in columns_to_keep_original]
    
    # 找出實際要保留的欄位
    columns_to_keep = []
    for clean_col, original_col in column_mapping.items():
        if clean_col in needed_columns_clean:
            columns_to_keep.append(original_col)
    
    # 顯示配對信息
    print(f"成功找到 {len(columns_to_keep)}/{len(columns_to_keep_original)} 個欄位")
    if len(columns_to_keep) < len(columns_to_keep_original):
        found_clean_cols = [clean_column_name(col) for col in columns_to_keep]
        missing_cols = [col for col, clean_col in zip(columns_to_keep_original, needed_columns_clean) if clean_col not in found_clean_cols]
        print(f"找不到以下欄位: {', '.join(missing_cols)}")
    
    # 如果資料中有標籤欄位，也保留它
    target_column = None
    for label_col in ['target', 'Bankrupt', 'Y', 'label']:
        if label_col in df.columns:
            if label_col not in columns_to_keep:
                columns_to_keep.append(label_col)
            print(f"已找到標籤欄位: {label_col}")
            target_column = label_col
            break
    
    #檢查是否有找到欄位
    if not columns_to_keep:
        print("警告: 沒有找到任何匹配的欄位!")
        print("原始數據的欄位名稱:")
        for col in df.columns:
            print(f"  - {col} (清理後: {clean_column_name(col)})")
        return None, None
    
    #選取需要的欄位
    filtered_df = df[columns_to_keep]
    
    return filtered_df, target_column

# 處理訓練集和測試集
filtered_train, train_target_column = process_dataset(df_train, "訓練")
filtered_test, test_target_column = process_dataset(df_test, "測試")

# 確保訓練集和測試集使用相同的特徵
if filtered_train is not None and filtered_test is not None:
    # 儲存原始測試集
    output_test_file = os.path.join(output_dir, "filtered_test_data.csv")
    filtered_test.to_csv(output_test_file, index=False)
    print(f"處理後的測試集已儲存至: {output_test_file}")
    print(f"測試集維度: {filtered_test.shape}")
    
    # 只對訓練集進行抽樣
    if train_target_column is not None:
        print(f"\n處理訓練集抽樣...")
        # 查看類別分佈
        value_counts = filtered_train[train_target_column].value_counts()
        print(f"目標欄位的分佈狀況：\n{value_counts}")
        print(f"不平衡比例：{value_counts.max() / value_counts.min():.2f}:1")

        # 分離特徵和目標變數
        X_train = filtered_train.drop(train_target_column, axis=1)
        y_train = filtered_train[train_target_column]
        
        # 儲存原始訓練集
        output_train_file = os.path.join(output_dir, "filtered_train_data_original.csv")
        filtered_train.to_csv(output_train_file, index=False)
        print(f"原始訓練集已儲存至: {output_train_file}")
        print(f"訓練集維度: {filtered_train.shape}")
        
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
