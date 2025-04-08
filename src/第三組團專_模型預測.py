import numpy as np
import pandas as pd
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import json


#載入最佳模型和閾值
def load_model_and_threshold(model_dir):
    """
    載入最佳模型和閾值
    
    Parameters:
    model_dir (str): 模型檔案所在目錄
    
    Returns:
    tuple: (最佳模型, 最佳閾值, 模型名稱, 模型參數)
    """
    # 尋找最佳模型資訊檔案
    model_info_path = os.path.join(model_dir, "best_model_info.json")
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r', encoding='utf-8') as file:
            best_model_info = json.load(file)
        
        model_name = best_model_info['model_name']
        best_threshold = best_model_info['threshold']
        model_params = best_model_info['model_parameters']
        
        # 找出模型檔案
        model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
        if not model_files:
            raise FileNotFoundError(f"在 {model_dir} 中找不到模型檔案")
        
        # 載入模型
        model_path = os.path.join(model_dir, model_files[0])
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    else:
        # 向下相容原有的載入方式
        # 找出模型檔案
        model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
        
        if not model_files:
            raise FileNotFoundError(f"在 {model_dir} 中找不到模型檔案")
        
        # 載入模型
        model_path = os.path.join(model_dir, model_files[0])
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        # 載入閾值資訊
        threshold_path = os.path.join(model_dir, "best_threshold_info.pkl")
        with open(threshold_path, 'rb') as file:
            threshold_info = pickle.load(file)
        
        best_threshold = threshold_info['best_threshold']
        model_name = threshold_info['model_name']
        model_params = {}  # 舊版不提供詳細參數
    
    print(f"已載入模型: {model_name}, 最佳閾值: {best_threshold:.2f}")
    print("模型參數:")
    for param_name, param_value in model_params.items():
        print(f"- {param_name}: {param_value}")
    
    return model, best_threshold, model_name, model_params

# 財務健康度評估
def assess_financial_health(model, data, feature_names, model_name):
    """
    評估財務健康度，並找出影響最大的三個欄位
    
    Parameters:
    model: 預訓練模型
    data (DataFrame): 要評估的資料
    feature_names (list): 特徵名稱列表
    model_name (str): 模型名稱
    
    Returns:
    tuple: (財務健康度分數, 影響最大的三個欄位及其重要性)
    """
    # 計算破產機率
    if hasattr(model, "predict_proba"):
        bankruptcy_prob = model.predict_proba(data)[:, 1]
    elif model_name == "神經網絡分類":
        bankruptcy_prob = model.predict(data).flatten()
    else:
        # 不支援機率輸出的模型，使用粗略估計
        bankruptcy_prob = model.predict(data).astype(float)
    
    # 計算財務健康度 (1 - 破產機率)
    financial_health = 1 - bankruptcy_prob
    
    # 獲取特徵重要性 (各模型處理方式不同)
    top_features = []
    if hasattr(model, 'feature_importances_'):
        # 決策樹類模型 (Random Forest, XGBoost, LightGBM等)
        importances = model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
        
        # 排序並取前三名
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:3]
    
    elif model_name == "SVC Classification" and hasattr(model, 'coef_'):
        # 線性SVM
        importances = np.abs(model.coef_[0])
        feature_importance = dict(zip(feature_names, importances))
        
        # 排序並取前三名
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:3]
    
    elif model_name == "神經網絡分類":
        # 神經網絡沒有直接特徵重要性，可以使用排列重要性或其他方法
        # 這裡僅提供一個替代方案，使用輸入層權重的絕對值
        try:
            first_layer_weights = np.abs(model.layers[0].get_weights()[0]).mean(axis=1)
            feature_importance = dict(zip(feature_names, first_layer_weights))
            
            # 排序並取前三名
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:3]
        except:
            top_features = [("無法確定", 0), ("無法確定", 0), ("無法確定", 0)]
    
    else:
        top_features = [("無法確定", 0), ("無法確定", 0), ("無法確定", 0)]
    
    return financial_health, top_features

#破產預測
def predict_bankruptcy(model, threshold, data, model_name):
    """
    使用模型進行破產預測
    
    Parameters:
    model: 預訓練模型
    threshold (float): 分類閾值
    data (DataFrame): 要預測的資料
    model_name (str): 模型名稱
    
    Returns:
    DataFrame: 包含預測結果的資料框
    """
    # 獲取預測概率
    if hasattr(model, "predict_proba"):
        pred_proba = model.predict_proba(data)[:, 1]
    elif model_name == "神經網絡分類":
        pred_proba = model.predict(data).flatten()
    else:
        # 不支援機率輸出的模型
        pred_class = model.predict(data)
        pred_proba = None
    
    # 根據閾值進行分類
    if pred_proba is not None:
        pred_class = (pred_proba >= threshold).astype(int)
    
    # 建立結果資料框
    results = pd.DataFrame()
    results['Bankruptcy_Prediction'] = pred_class
    
    # 如果有概率值，則添加到結果中
    if pred_proba is not None:
        results['Bankruptcy_Probability'] = pred_proba
    
    return results

def save_predictions_with_check(result_df, base_output_file):
    """
    安全保存預測結果，處理檔案已存在的情況
    
    Parameters:
    result_df (DataFrame): 包含預測結果的資料框
    base_output_file (str): 基本輸出檔案路徑
    
    Returns:
    str: 實際保存的檔案路徑
    """
    import os
    from datetime import datetime
    
    # 準備輸出檔案名稱
    output_dir = os.path.dirname(base_output_file)
    base_name = os.path.basename(base_output_file)
    name_parts = os.path.splitext(base_name)
    
    # 檢查原始檔案是否存在
    if os.path.exists(base_output_file):
        # 如果存在，使用時間戳建立新檔名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{name_parts[0]}_{timestamp}{name_parts[1]}"
        output_file = os.path.join(output_dir, new_filename)
        print(f"檔案 {base_output_file} 已存在，使用新檔名: {output_file}")
    else:
        output_file = base_output_file
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 嘗試保存檔案
    try:
        result_df.to_csv(output_file, index=False)
        print(f"預測結果已保存至: {output_file}")
        return output_file
    except PermissionError:
        # 如果仍有權限問題，使用目前目錄作為備用
        fallback_file = os.path.join(os.getcwd(), f"predictions_{name_parts[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        print(f"無法寫入 {output_file}，改為保存至: {fallback_file}")
        result_df.to_csv(fallback_file, index=False)
        return fallback_file


def main():
    # 設定模型目錄
    model_dir = "D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/models/64/ada"
    
    # 檢查命令行參數，看是否提供了數據文件
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/data/64/filtered_test_data.csv"
    
    try:
        # 載入模型和閾值
        model, threshold, model_name, model_params = load_model_and_threshold(model_dir)
        
        # 載入要預測的數據
        print(f"載入資料: {data_file}")
        data = pd.read_csv(data_file)
        
        # 儲存特徵名稱
        feature_names = data.columns.tolist()
        if "Bankrupt" in feature_names:
            feature_names.remove("Bankrupt")
            y_actual = data["Bankrupt"]
            data_features = data.drop("Bankrupt", axis=1)
            has_actual = True
        else:
            data_features = data
            has_actual = False

        # 檢查是否包含目標變數列，如果有則移除
        if "Bankrupt" in data.columns:
            # 儲存實際值以便後續對比
            y_actual = data["Bankrupt"]
            data = data.drop("Bankrupt", axis=1)
            has_actual = True
        else:
            has_actual = False
        
        # 執行預測
        print("執行預測...")
        predictions = predict_bankruptcy(model, threshold, data, model_name)

        # 評估財務健康度
        print("評估財務健康度...")
        financial_health, top_features = assess_financial_health(model, data_features, feature_names, model_name)

        # 將財務健康度添加到結果中
        predictions['Financial_Health_Score'] = financial_health
        
        # 合併預測結果與原始資料
        result_df = pd.concat([data_features, predictions], axis=1)
        
        # 如果有實際值，添加到結果中
        if has_actual:
            result_df['Actual_Bankruptcy'] = y_actual
        
        # 保存預測結果
        base_output_file = os.path.splitext(data_file)[0] + "_predictions.csv"
        output_file = save_predictions_with_check(result_df, base_output_file)
        print(f"預測結果已保存至: {output_file}")
        
        # 顯示財務健康度評估結果
        print("\n財務健康度評估結果:")
        print("-" * 50)
        
        # 顯示影響最大的三個欄位
        print("\n影響財務健康度最大的三個因素:")
        for feature, importance in top_features:
            print(f"- {feature}: 重要性 = {importance:.4f}")

        # 如果有實際值，計算並顯示簡單的評估指標
        metrics = None
        if has_actual:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            y_pred = predictions['Bankruptcy_Prediction']
            accuracy = accuracy_score(y_actual, y_pred)
            precision = precision_score(y_actual, y_pred)
            recall = recall_score(y_actual, y_pred)
            f1 = f1_score(y_actual, y_pred)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print("\n預測評估結果:")
            print(f"準確率: {accuracy:.4f}")
            print(f"精確率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分數: {f1:.4f}")
        
        
    except Exception as e:
        print(f"執行時發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()