import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from functools import partial
import time
import os
import pickle
import json
import sys
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import ParameterGrid
import xgboost as xgb
import lightgbm as lgb
import traceback
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# 在主函數開始處添加
print(f"Python版本: {sys.version}")
print(f"執行路徑: {sys.executable}")

# 嘗試顯示更多GPU信息
try:
    print("\n=== NVIDIA-SMI 信息 ===")
    import subprocess
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))
except:
    print("無法獲取NVIDIA-SMI信息")

# 在導入模組後添加 GPU_AVAILABLE 的定義
def check_gpu_environment():
    """檢查GPU環境並輸出診斷信息"""
    print("\n=== GPU環境檢查 ===")
    
    gpu_available = False
    
    try:
        import xgboost as xgb
        print(f"XGBoost版本: {xgb.__version__}")
        has_gpu = xgb.config.build_info().get('USE_CUDA', '0') == '1'
        print(f"XGBoost是否支持GPU: {'是' if has_gpu else '否'}")
        if has_gpu:
            gpu_available = True
    except:
        print("無法檢查XGBoost GPU支持")
    
    try:
        import lightgbm as lgb
        print(f"LightGBM版本: {lgb.__version__}")
        # 如果 LightGBM 支援 GPU，也設置為 True
        if hasattr(lgb, 'LGBMModel'):
            gpu_available = True
    except:
        print("無法檢查LightGBM版本")
    
    print("=== GPU環境檢查完成 ===\n")
    return gpu_available

# 在主程式開始前定義全局變數
GPU_AVAILABLE = check_gpu_environment()

# 在程式開始處添加
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用 GPU

# 簡化評估函數
def evaluate_model(model, X, y, model_name):
    """評估模型性能"""
    # 獲取預測結果
    y_pred = model.predict(X)
    
    # 計算各項指標
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # 計算 AUC
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.decision_function(X)
    auc_score = roc_auc_score(y, y_prob)
    
    # 尋找最佳閾值
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # 返回結果字典，加入 model 對象
    return {
        'model_name': model_name,
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'threshold': optimal_threshold
    }

# 簡化模型訓練函數
def train_model(model_class, X_train, y_train, params=None, model_name="", gpu_params=None):
    """通用模型訓練函數"""
    if gpu_params and GPU_AVAILABLE:
        params = {**(params or {}), **gpu_params}
        
    model = model_class(**(params or {}))
    print(f"{model_name}訓練中...")
    model.fit(X_train, y_train)
    print(f"{model_name}訓練完成")
    return model

# 使用k折交叉驗證訓練和評估模型
def train_with_cross_validation(X, y, model_func, params=None, cv=5, scoring='f1'):
    """
    使用k折交叉驗證訓練和評估模型
    
    參數:
        X: 特徵資料
        y: 標籤
        model_func: 模型訓練函數
        params: 模型參數
        cv: 折數
        scoring: 評分方式
    
    返回:
        scores: 各折分數
        mean_score: 平均分數
        std_score: 標準差
    """
    # 使用分層k折交叉驗證以保持每折中目標變數的比例
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    print(f"開始{cv}折交叉驗證...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"訓練第{fold+1}折...")
        model = model_func(X_train_fold, y_train_fold, params)
        
        # 評估模型
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_val_fold)
        
        # 計算指定的評分
        if scoring == 'f1':
            score = f1_score(y_val_fold, y_pred)
        elif scoring == 'accuracy':
            score = accuracy_score(y_val_fold, y_pred)
        elif scoring == 'precision':
            score = precision_score(y_val_fold, y_pred)
        elif scoring == 'recall':
            score = recall_score(y_val_fold, y_pred)
        elif scoring == 'roc_auc' and hasattr(model, "predict_proba"):
            score = roc_auc_score(y_val_fold, y_pred_proba)
        else:
            score = f1_score(y_val_fold, y_pred)
        
        scores.append(score)
        print(f"第{fold+1}折 {scoring}分數: {score:.4f}")
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"{cv}折交叉驗證平均{scoring}分數: {mean_score:.4f} (±{std_score:.4f})")
    
    return scores, mean_score, std_score

# 修改XGBoost訓練函數
def train_xgboost(X_train, y_train, params=None):
    default_params = {
        'scale_pos_weight': 10,
        'learning_rate': 0.05,
        'max_depth': 4,
        'min_child_weight': 2,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 200,
        'tree_method': 'hist',
        'enable_categorical': False,
        'verbosity': 1  # 添加詳細輸出
    }

    if params:
        default_params.update(params)
    
    print("\nXGBoost訓練開始...")
    print("使用的參數:", default_params)

    model = XGBClassifier(**default_params)
    
    # 使用 tqdm 顯示訓練進度
    with tqdm(total=1, desc="XGBoost訓練進度") as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)
    
    print("XGBoost訓練完成!")
    return model

# 修改LightGBM訓練函數
def train_lightgbm(X_train, y_train, params=None):
    default_params = {
        'class_weight': 'balanced',
        'learning_rate': 0.05,
        'n_estimators': 200,
        'num_leaves': 31,
        'max_depth': 5,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1
    }

    if params:
        default_params.update(params)
        
        print("\nLightGBM訓練開始...")
        print("使用的參數:", default_params)

    model = LGBMClassifier(**default_params)
    
    # 使用 tqdm 顯示訓練進度
    with tqdm(total=1, desc="LightGBM訓練進度") as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)
    
    print("LightGBM訓練完成!")
    return model

# 測試不同閾值並找出最佳的
def find_optimal_threshold(model, X_test, y_test, model_name, start=0.1, end=0.9, step=0.1):
    thresholds = np.arange(start, end + step, step)
    results = []

    for threshold in thresholds:
        result = evaluate_model(model, X_test, y_test, model_name)
        result['threshold'] = threshold
        results.append(result)

    # 根據F1分數找出最佳閾值
    best_result = max(results, key=lambda x: x['f1'])

    print(f"\n最佳閾值（基於F1分數）: {best_result['threshold']:.2f}")
    print(f"對應的F1分數: {best_result['f1']:.4f}")

    return results, best_result

# 繪製分類模型比較圖表
def plot_classification_comparison(results, dataset_name=""):
    """繪製三組資料的分類評估指標比較圖"""
    plt.figure(figsize=(12, 6))
    
    models = [result['model_name'] for result in results]
    metrics = {
        'Accuracy': [result['accuracy'] for result in results],
        'Precision': [result['precision'] for result in results],
        'Recall': [result['recall'] for result in results],
        'F1': [result['f1'] for result in results]
    }
    
    x = np.arange(len(models))
    width = 0.2
    multiplier = 0
    
    for metric, scores in metrics.items():
        offset = width * multiplier
        plt.bar(x + offset, scores, width, label=metric)
        multiplier += 1
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title(f'{dataset_name} - Comparison of Evaluation Results')
    plt.xticks(x + width * 1.5, models, rotation=45)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # 保存圖片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/models/35/ori/comparison_{dataset_name}_{timestamp}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

# 繪製ROC曲線
def plot_roc_curves(results, X, y, dataset_name=""):
    """繪製 ROC 曲線"""
    plt.figure(figsize=(10, 8))

    for result in results:
        model = result['model']
        model_name = result['model_name']
        
        # 獲取預測概率
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = model.decision_function(X)

        # 計算 ROC 曲線
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        # 繪製 ROC 曲線
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} - ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    

# 修改特徵重要性分析函數
def analyze_feature_importance(models, feature_names, output_dir):
    print("\n分析特徵重要性...")
    model_importances = {}
    
    # 收集所有支援特徵重要性的模型的結果
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(
                model.feature_importances_, 
                index=feature_names
            ).sort_values(ascending=True)  # 改為升序以便找出最不重要的特徵
            model_importances[model_name] = importances
    
    # 如果有模型支援特徵重要性分析
    if model_importances:
        # 計算平均特徵重要性
        all_importances = pd.DataFrame(model_importances)
        mean_importance = all_importances.mean(axis=1).sort_values(ascending=True)
        
        # 準備輸出文件
        output_file = os.path.join(output_dir, "feature_importance_analysis.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 寫入20個最不相關的特徵
            f.write("=== 20 most irrelevant features ===\n\n")
            for rank, (feature, score) in enumerate(mean_importance[:20].items(), 1):
                f.write(f"{rank} : {feature}\n")
                f.write(f"score: {score:.6f}\n\n")
            
            # 寫入剩餘特徵的重要性
            f.write("\n=== The importance ranking of the remaining features ===\n\n")
            for rank, (feature, score) in enumerate(mean_importance[20:].items(), 21):
                f.write(f"{rank} : {feature}\n")
                f.write(f"score: {score:.6f}\n\n")
        
        print(f"特徵重要性分析結果已保存至: {output_file}")
        
        # 繪製特徵重要性圖表
        plt.figure(figsize=(12, 6))
        mean_importance.head(20).plot(kind='barh')
        plt.title('The 20 Most Irrelevant Features')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        # 保存特徵重要性圖表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{output_dir}/feature_importance_{timestamp}.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    return model_importances

# 修改彙總結果的函數，加入交叉驗證結果
def summarize_model_results(results, dataset_type="Test"):
    print("\n" + "="*60)
    print(f"所有模型在{dataset_type}集上的評估結果彙總")
    print("="*60)
    
    # 創建結果列表來存儲所有指標
    summary_list = []
    
    for result in results:
        model_name = result['model_name']
        print(f"\n{model_name} 評估結果:")
        print(f"準確率: {result['accuracy']:.4f}")
        print(f"精確率: {result['precision']:.4f}")
        print(f"召回率: {result['recall']:.4f}")
        print(f"F1分數: {result['f1']:.4f}")
        if result['auc'] is not None:
            print(f"AUC: {result['auc']:.4f}")
        print(f"最佳閾值: {result['threshold']:.2f}")
        
        # 如果有交叉驗證結果，則顯示
        cv_mean = None
        cv_std = None
        if 'cv_scores' in result:
            cv_mean = result['cv_scores']['mean']
            cv_std = result['cv_scores']['std']
            print(f"交叉驗證F1分數: {cv_mean:.4f} (±{cv_std:.4f})")
        
        # 收集該模型的所有指標
        model_summary = {
            '數據集': dataset_type,
            '模型': model_name,
            '準確率': result['accuracy'],
            '精確率': result['precision'],
            '召回率': result['recall'],
            'F1分數': result['f1'],
            'AUC': result['auc'] if result['auc'] is not None else None,
            '最佳閾值': result['threshold']
        }
        
        # 如果有交叉驗證結果，添加到摘要中
        if cv_mean is not None:
            model_summary['交叉驗證F1平均值'] = cv_mean
            model_summary['交叉驗證F1標準差'] = cv_std
            
        summary_list.append(model_summary)
        print("-"*60)
    
    # 找出最佳模型 (基於F1分數)
    best_result = max(results, key=lambda x: x['f1'])
    print(f"\n最佳模型是: {best_result['model_name']}")
    print(f"F1分數: {best_result['f1']:.4f}")
    print(f"最佳閾值: {best_result['threshold']:.2f}")
    if 'cv_scores' in best_result:
        print(f"交叉驗證F1分數: {best_result['cv_scores']['mean']:.4f} (±{best_result['cv_scores']['std']:.4f})")
    print("="*60)
    
    return summary_list

# 為XGBoost添加網格搜索功能
def find_best_xgboost_params(X_train, y_train, X_val, y_val):
    print("開始 XGBoost 參數搜尋...")
    param_grid = {
        'max_depth': [3, 5],           
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100],         
        'min_child_weight': [1],       
        'gamma': [0],                  
        'subsample': [0.8],            
        'colsample_bytree': [0.8],     
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 1.0]
    }
    
    search_results = []
    param_combinations = list(ParameterGrid(param_grid))
    total_combinations = len(param_combinations)
    
    print(f"總共需要測試 {total_combinations} 種參數組合")
    
    best_f1 = 0
    best_params = None
    best_threshold = None
    best_model = None
    
    try:
        for i, params in enumerate(param_combinations, 1):
            try:
                start_time = time.time()
                
                # 設定模型參數
                model = xgb.XGBClassifier(
                    **params,
                    random_state=42,
                    eval_metric='auc'
                )
                
                # 訓練模型
                model.fit(
                    X_train, 
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # 獲取預測概率
                y_prob = model.predict_proba(X_val)[:, 1]
                
                # 計算在不同閾值下的 F1 分數
                thresholds = np.arange(0.1, 0.91, 0.1)
                f1_scores = []
                
                for threshold in thresholds:
                    y_pred = (y_prob >= threshold).astype(int)
                    f1 = f1_score(y_val, y_pred)
                    f1_scores.append((threshold, f1))
                
                # 找出最佳閾值
                best_threshold_tuple = max(f1_scores, key=lambda x: x[1])
                optimal_threshold = best_threshold_tuple[0]
                f1 = best_threshold_tuple[1]
                
                # 使用最佳閾值進行預測
                y_pred = (y_prob >= optimal_threshold).astype(int)
                accuracy = accuracy_score(y_val, y_pred)
                
                # 輸出詳細的預測分布資訊，以便診斷
                positive_count = np.sum(y_pred)
                total_count = len(y_pred)
                print(f"閾值: {optimal_threshold}, 預測為正例的比例: {positive_count}/{total_count} ({positive_count/total_count:.2%})")
                print(f"實際正例比例: {np.sum(y_val)}/{len(y_val)} ({np.sum(y_val)/len(y_val):.2%})")
                
                training_time = time.time() - start_time
                
                result = {
                    '參數組合': str(params),
                    'F1分數': f1,
                    '準確率': accuracy,
                    '訓練時間': training_time,
                    '最佳閾值': optimal_threshold
                }
                result.update(params)
                search_results.append(result)
                
                # 更新最佳模型
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = params
                    best_threshold = optimal_threshold
                    best_model = model
                    print(f"找到新的最佳模型: F1={best_f1:.4f}, 閾值={best_threshold:.2f}")
                
                print(f"進度: {i}/{total_combinations}, F1: {f1:.4f}, 閾值: {optimal_threshold:.2f}, 時間: {training_time:.2f}秒")
                
            except Exception as e:
                print(f"參數組合 {params} 發生錯誤: {str(e)}")
                continue
                
    except KeyboardInterrupt:
        print("\n搜尋被使用者中斷")
    
    # 輸出最佳結果
    print("\nXGBoost 最佳結果:")
    if best_params is not None:
        print(f"最佳參數: {best_params}")
    else:
        print("未找到最佳參數")
        
    if best_f1 != 0:
        print(f"最佳F1分數: {best_f1:.4f}")
    else:
        print("未找到有效的F1分數")
        
    if best_threshold is not None:
        print(f"最佳閾值: {best_threshold:.4f}")
    else:
        print("未找到最佳閾值")
    
    # 額外驗證最佳模型表現
    if best_model is not None:
        print("\n驗證最佳模型表現:")
        y_prob = best_model.predict_proba(X_val)[:, 1]
        
        # 在不同閾值下的表現
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            print(f"閾值={threshold:.2f}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    return search_results, best_model, best_params, best_threshold

# 為LightGBM添加網格搜索功能
def find_best_lightgbm_params(X_train, y_train, X_val, y_val):
    print("開始 LightGBM 參數搜尋...")
    param_grid = {
        'num_leaves': [31],            
        'max_depth': [3, 5],           
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100],         
        'min_child_samples': [20],     
        'subsample': [0.8],            
        'colsample_bytree': [0.8],     
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 1.0]
    }
    
    search_results = []
    param_combinations = list(ParameterGrid(param_grid))
    total_combinations = len(param_combinations)
    
    print(f"總共需要測試 {total_combinations} 種參數組合")
    
    best_f1 = 0
    best_params = None
    best_threshold = None
    best_model = None
    
    try:
        for i, params in enumerate(param_combinations, 1):
            try:
                start_time = time.time()
                
                model = lgb.LGBMClassifier(**params, random_state=42)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc')
                
                # 計算最佳閾值
                y_prob = model.predict_proba(X_val)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_val, y_prob)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                
                # 使用最佳閾值進行預測
                y_pred = (y_prob >= optimal_threshold).astype(int)
                f1 = f1_score(y_val, y_pred)
                accuracy = accuracy_score(y_val, y_pred)
                
                training_time = time.time() - start_time
                
                result = {
                    '參數組合': str(params),
                    'F1分數': f1,
                    '準確率': accuracy,
                    '訓練時間': training_time,
                    '最佳閾值': optimal_threshold
                }
                result.update(params)
                search_results.append(result)
                
                # 更新最佳模型
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = params
                    best_threshold = optimal_threshold
                    best_model = model
                
                print(f"進度: {i}/{total_combinations}, F1: {f1:.4f}, 時間: {training_time:.2f}秒")
                
            except Exception as e:
                print(f"參數組合 {params} 發生錯誤: {str(e)}")
                continue
                
    except KeyboardInterrupt:
        print("\n搜尋被使用者中斷")
    
    print("\nLightGBM 最佳結果:")
    if best_params is not None:
        print(f"最佳參數: {best_params}")
    else:
        print("未找到最佳參數")
        
    if best_f1 != 0:
        print(f"最佳F1分數: {best_f1:.4f}")
    else:
        print("未找到有效的F1分數")
        
    if best_threshold is not None:
        print(f"最佳閾值: {best_threshold:.4f}")
    else:
        print("未找到最佳閾值")
    
    return search_results, best_model, best_params, best_threshold

# 為Random Forest添加網格搜索功能
def find_best_random_forest_params(X_train, y_train, X_val, y_val):
    print("開始 Random Forest 參數搜尋...")
    # 減少參數空間以提高效率
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],  # 移除None以減少計算時間
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']  # 移除None以減少計算時間
    }
    
    search_results = []
    param_combinations = list(ParameterGrid(param_grid))
    total_combinations = len(param_combinations)
    
    print(f"總共需要測試 {total_combinations} 種參數組合")
    
    best_f1 = 0
    best_params = None
    best_threshold = None
    best_model = None
    
    try:
        for i, params in enumerate(param_combinations, 1):
            try:
                start_time = time.time()
                
                # 設定模型參數
                model = RandomForestClassifier(
                    **params,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                )
                
                # 訓練模型
                model.fit(X_train, y_train)
                
                # 獲取預測概率
                y_prob = model.predict_proba(X_val)[:, 1]
                
                # 計算在不同閾值下的 F1 分數
                thresholds = np.arange(0.1, 0.91, 0.1)
                f1_scores = []
                
                for threshold in thresholds:
                    y_pred = (y_prob >= threshold).astype(int)
                    f1 = f1_score(y_val, y_pred)
                    f1_scores.append((threshold, f1))
                
                # 找出最佳閾值
                best_threshold_tuple = max(f1_scores, key=lambda x: x[1])
                optimal_threshold = best_threshold_tuple[0]
                f1 = best_threshold_tuple[1]
                
                # 使用最佳閾值進行預測
                y_pred = (y_prob >= optimal_threshold).astype(int)
                accuracy = accuracy_score(y_val, y_pred)
                
                training_time = time.time() - start_time
                
                result = {
                    '參數組合': str(params),
                    'F1分數': f1,
                    '準確率': accuracy,
                    '訓練時間': training_time,
                    '最佳閾值': optimal_threshold
                }
                result.update(params)
                search_results.append(result)
                
                # 更新最佳模型
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = params
                    best_threshold = optimal_threshold
                    best_model = model
                    print(f"找到新的最佳模型: F1={best_f1:.4f}, 閾值={best_threshold:.2f}")
                
                print(f"進度: {i}/{total_combinations}, F1: {f1:.4f}, 閾值: {optimal_threshold:.2f}, 時間: {training_time:.2f}秒")
                
            except Exception as e:
                print(f"參數組合 {params} 發生錯誤: {str(e)}")
                continue
                
    except KeyboardInterrupt:
        print("\n搜尋被使用者中斷")
    
    # 輸出最佳結果
    print("\nRandom Forest 最佳結果:")
    if best_params is not None:
        print(f"最佳參數: {best_params}")
    else:
        print("未找到最佳參數")
        
    if best_f1 != 0:
        print(f"最佳F1分數: {best_f1:.4f}")
    else:
        print("未找到有效的F1分數")
        
    if best_threshold is not None:
        print(f"最佳閾值: {best_threshold:.4f}")
    else:
        print("未找到最佳閾值")
    
    return search_results, best_model, best_params, best_threshold

# 為SVM添加網格搜索功能
def find_best_svm_params(X_train, y_train, X_val, y_val):
    print("開始 SVM 參數搜尋...")
    # 減少參數空間以提高效率
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear'],  # 移除poly以減少計算時間
        'gamma': ['scale', 'auto']  # 減少gamma選項
    }
    
    search_results = []
    param_combinations = list(ParameterGrid(param_grid))
    total_combinations = len(param_combinations)
    
    print(f"總共需要測試 {total_combinations} 種參數組合")
    
    best_f1 = 0
    best_params = None
    best_threshold = None
    best_model = None
    
    try:
        for i, params in enumerate(param_combinations, 1):
            try:
                start_time = time.time()
                
                # 設定模型參數
                model = SVC(
                    **params,
                    probability=True,
                    random_state=42,
                    class_weight='balanced'
                )
                
                # 訓練模型
                model.fit(X_train, y_train)
                
                # 獲取預測概率
                y_prob = model.predict_proba(X_val)[:, 1]
                
                # 計算在不同閾值下的 F1 分數
                thresholds = np.arange(0.1, 0.91, 0.1)
                f1_scores = []
                
                for threshold in thresholds:
                    y_pred = (y_prob >= threshold).astype(int)
                    f1 = f1_score(y_val, y_pred)
                    f1_scores.append((threshold, f1))
                
                # 找出最佳閾值
                best_threshold_tuple = max(f1_scores, key=lambda x: x[1])
                optimal_threshold = best_threshold_tuple[0]
                f1 = best_threshold_tuple[1]
                
                # 使用最佳閾值進行預測
                y_pred = (y_prob >= optimal_threshold).astype(int)
                accuracy = accuracy_score(y_val, y_pred)
                
                training_time = time.time() - start_time
                
                result = {
                    '參數組合': str(params),
                    'F1分數': f1,
                    '準確率': accuracy,
                    '訓練時間': training_time,
                    '最佳閾值': optimal_threshold
                }
                result.update(params)
                search_results.append(result)
                
                # 更新最佳模型
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = params
                    best_threshold = optimal_threshold
                    best_model = model
                    print(f"找到新的最佳模型: F1={best_f1:.4f}, 閾值={best_threshold:.2f}")
                
                print(f"進度: {i}/{total_combinations}, F1: {f1:.4f}, 閾值: {optimal_threshold:.2f}, 時間: {training_time:.2f}秒")
                
            except Exception as e:
                print(f"參數組合 {params} 發生錯誤: {str(e)}")
                continue
                
    except KeyboardInterrupt:
        print("\n搜尋被使用者中斷")
    
    # 輸出最佳結果
    print("\nSVM 最佳結果:")
    if best_params is not None:
        print(f"最佳參數: {best_params}")
    else:
        print("未找到最佳參數")
        
    if best_f1 != 0:
        print(f"最佳F1分數: {best_f1:.4f}")
    else:
        print("未找到有效的F1分數")
        
    if best_threshold is not None:
        print(f"最佳閾值: {best_threshold:.4f}")
    else:
        print("未找到最佳閾值")
    
    return search_results, best_model, best_params, best_threshold

# 為神經網絡添加網格搜索功能
def find_best_neural_network_params(X_train, y_train, X_val, y_val):
    print("開始 神經網絡 參數搜尋...")
    # 減少參數空間以提高效率
    param_grid = {
        'hidden_layer_sizes': [(64,), (128,), (64, 32)],  # 減少層數組合
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],  # 只使用adam優化器
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01]
    }
    
    search_results = []
    param_combinations = list(ParameterGrid(param_grid))
    total_combinations = len(param_combinations)
    
    print(f"總共需要測試 {total_combinations} 種參數組合")
    
    best_f1 = 0
    best_params = None
    best_threshold = None
    best_model = None
    
    try:
        for i, params in enumerate(param_combinations, 1):
            try:
                start_time = time.time()
                
                # 設定模型參數
                model = MLPClassifier(
                    **params,
                    max_iter=200,
                    early_stopping=True,
                    random_state=42
                )
                
                # 訓練模型
                model.fit(X_train, y_train)
                
                # 獲取預測概率
                y_prob = model.predict_proba(X_val)[:, 1]
                
                # 計算在不同閾值下的 F1 分數
                thresholds = np.arange(0.1, 0.91, 0.1)
                f1_scores = []
                
                for threshold in thresholds:
                    y_pred = (y_prob >= threshold).astype(int)
                    f1 = f1_score(y_val, y_pred)
                    f1_scores.append((threshold, f1))
                
                # 找出最佳閾值
                best_threshold_tuple = max(f1_scores, key=lambda x: x[1])
                optimal_threshold = best_threshold_tuple[0]
                f1 = best_threshold_tuple[1]
                
                # 使用最佳閾值進行預測
                y_pred = (y_prob >= optimal_threshold).astype(int)
                accuracy = accuracy_score(y_val, y_pred)
                
                training_time = time.time() - start_time
                
                result = {
                    '參數組合': str(params),
                    'F1分數': f1,
                    '準確率': accuracy,
                    '訓練時間': training_time,
                    '最佳閾值': optimal_threshold
                }
                result.update({k: str(v) if k == 'hidden_layer_sizes' else v for k, v in params.items()})
                search_results.append(result)
                
                # 更新最佳模型
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = params
                    best_threshold = optimal_threshold
                    best_model = model
                    print(f"找到新的最佳模型: F1={best_f1:.4f}, 閾值={best_threshold:.2f}")
                
                print(f"進度: {i}/{total_combinations}, F1: {f1:.4f}, 閾值: {optimal_threshold:.2f}, 時間: {training_time:.2f}秒")
                
            except Exception as e:
                print(f"參數組合 {params} 發生錯誤: {str(e)}")
                continue
                
    except KeyboardInterrupt:
        print("\n搜尋被使用者中斷")
    
    # 輸出最佳結果
    print("\n神經網絡最佳結果:")
    if best_params is not None:
        print(f"最佳參數: {best_params}")
    else:
        print("未找到最佳參數")
        
    if best_f1 != 0:
        print(f"最佳F1分數: {best_f1:.4f}")
    else:
        print("未找到有效的F1分數")
        
    if best_threshold is not None:
        print(f"最佳閾值: {best_threshold:.4f}")
    else:
        print("未找到最佳閾值")
    
    return search_results, best_model, best_params, best_threshold

# 修改主程式來使用新的網格搜索功能
def main():
    try:
        # 設定輸出路徑
        output_path = "D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/models/35/ori"
        
        # 載入數據
        print("\n1. 載入數據")
        train_data = pd.read_csv("D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/data/35/done/se/filtered_train_data_original.csv")
        val_data = pd.read_csv("D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/data/35/done/se/filtered_val_data.csv")
        test_data = pd.read_csv("D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/data/35/done/se/filtered_test_data.csv")
        
        # 準備特徵和標籤
        X_train = train_data.drop("Bankrupt", axis=1)
        y_train = train_data["Bankrupt"]
        X_val = val_data.drop("Bankrupt", axis=1)
        y_val = val_data["Bankrupt"]
        X_test = test_data.drop("Bankrupt", axis=1)
        y_test = test_data["Bankrupt"]
        
        # 標準化特徵以用於SVM和神經網絡
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 初始化參數搜尋結果字典
        param_search_results = {}
        
        # XGBoost 參數搜尋
        print("\n2. 執行 XGBoost 參數搜尋")
        xgboost_results, xgb_best_model, xgb_best_params, xgb_best_threshold = find_best_xgboost_params(X_train, y_train, X_val, y_val)
        param_search_results['XGBoost'] = xgboost_results
        
        # LightGBM 參數搜尋
        print("\n3. 執行 LightGBM 參數搜尋")
        lightgbm_results, lgb_best_model, lgb_best_params, lgb_best_threshold = find_best_lightgbm_params(X_train, y_train, X_val, y_val)
        param_search_results['LightGBM'] = lightgbm_results
        
        # Random Forest 參數搜尋
        print("\n4. 執行 Random Forest 參數搜尋")
        rf_results, rf_best_model, rf_best_params, rf_best_threshold = find_best_random_forest_params(X_train, y_train, X_val, y_val)
        param_search_results['Random Forest'] = rf_results
        
        # SVM 參數搜尋 (使用標準化特徵)
        print("\n5. 執行 SVM 參數搜尋")
        svm_results, svm_best_model, svm_best_params, svm_best_threshold = find_best_svm_params(X_train_scaled, y_train, X_val_scaled, y_val)
        param_search_results['SVM'] = svm_results
        
        # 神經網絡參數搜尋 (使用標準化特徵)
        print("\n6. 執行神經網絡參數搜尋")
        nn_results, nn_best_model, nn_best_params, nn_best_threshold = find_best_neural_network_params(X_train_scaled, y_train, X_val_scaled, y_val)
        param_search_results['Neural Network'] = nn_results
        
        # 建立模型字典
        models = {
            'XGBoost': xgb_best_model,
            'LightGBM': lgb_best_model,
            'Random Forest': rf_best_model,
            'SVM': svm_best_model,
            'Neural Network': nn_best_model
        }
        
        # 評估並繪製圖表
        print("\n7. 評估結果")
        
        # 訓練集評估
        train_results = []
        for model_name, model in models.items():
            # 針對SVM和神經網絡使用標準化的數據
            if model_name in ['SVM', 'Neural Network']:
                train_result = evaluate_model(model, X_train_scaled, y_train, model_name)
            else:
                train_result = evaluate_model(model, X_train, y_train, model_name)
            train_results.append(train_result)
        print("\n=== 訓練集評估結果 ===")
        train_summaries = summarize_model_results(train_results, "Training")
        
        # 驗證集評估
        val_results = []
        for model_name, model in models.items():
            # 針對SVM和神經網絡使用標準化的數據
            if model_name in ['SVM', 'Neural Network']:
                val_result = evaluate_model(model, X_val_scaled, y_val, model_name)
            else:
                val_result = evaluate_model(model, X_val, y_val, model_name)
            val_results.append(val_result)
        print("\n=== 驗證集評估結果 ===")
        val_summaries = summarize_model_results(val_results, "Validation")
        
        # 測試集評估
        test_results = []
        for model_name, model in models.items():
            # 針對SVM和神經網絡使用標準化的數據
            if model_name in ['SVM', 'Neural Network']:
                test_result = evaluate_model(model, X_test_scaled, y_test, model_name)
            else:
                test_result = evaluate_model(model, X_test, y_test, model_name)
            test_results.append(test_result)
        print("\n=== 測試集評估結果 ===")
        test_summaries = summarize_model_results(test_results, "Test")
        
        # 繪製合併的分類指標比較圖
        plot_combined_classification_comparison(train_results, val_results, test_results, output_path)
        
        # 繪製合併的ROC曲線
        plot_combined_roc_curves(train_results, val_results, test_results, 
                             X_train, y_train, X_val, y_val, X_test, y_test, 
                             models, X_train_scaled, X_val_scaled, X_test_scaled,
                             output_path)  # 添加 output_path 參數
        
        # 繪製F1分數比較圖
        plot_f1_comparison(train_results, val_results, test_results, output_path)
        
        # 比較兩個模型的最佳結果
        print("\n=== 最佳模型比較 (僅考慮XGBoost和LightGBM) ===")
        
        # XGBoost 評估
        if xgb_best_model is not None and xgb_best_threshold is not None:
            y_prob = xgb_best_model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= xgb_best_threshold).astype(int)
            xgb_f1 = f1_score(y_val, y_pred)
            xgb_precision = precision_score(y_val, y_pred)
            xgb_recall = recall_score(y_val, y_pred)
            
            print("\nXGBoost 詳細驗證結果:")
            print(f"最佳參數: {xgb_best_params}")
            print(f"最佳閾值: {xgb_best_threshold:.4f}")
            print(f"驗證集 F1 分數: {xgb_f1:.4f}")
            print(f"驗證集 Precision: {xgb_precision:.4f}")
            print(f"驗證集 Recall: {xgb_recall:.4f}")
            
            # 檢查預測分布
            positives = np.sum(y_pred)
            total = len(y_pred)
            print(f"預測正例數: {positives}/{total} ({positives/total:.2%})")
            print(f"實際正例數: {np.sum(y_val)}/{len(y_val)} ({np.sum(y_val)/len(y_val):.2%})")
        else:
            print("\nXGBoost 模型訓練失敗")
            xgb_f1 = 0
            
        # LightGBM 評估
        if lgb_best_model is not None and lgb_best_threshold is not None:
            y_prob = lgb_best_model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= lgb_best_threshold).astype(int)
            lgb_f1 = f1_score(y_val, y_pred)
            lgb_precision = precision_score(y_val, y_pred)
            lgb_recall = recall_score(y_val, y_pred)
            
            print("\nLightGBM 詳細驗證結果:")
            print(f"最佳參數: {lgb_best_params}")
            print(f"最佳閾值: {lgb_best_threshold:.4f}")
            print(f"驗證集 F1 分數: {lgb_f1:.4f}")
            print(f"驗證集 Precision: {lgb_precision:.4f}")
            print(f"驗證集 Recall: {lgb_recall:.4f}")
            
            # 檢查預測分布
            positives = np.sum(y_pred)
            total = len(y_pred)
            print(f"預測正例數: {positives}/{total} ({positives/total:.2%})")
            print(f"實際正例數: {np.sum(y_val)}/{len(y_val)} ({np.sum(y_val)/len(y_val):.2%})")
        else:
            print("\nLightGBM 模型訓練失敗")
            lgb_f1 = 0
        
        # Random Forest 評估
        if rf_best_model is not None and rf_best_threshold is not None:
            y_prob = rf_best_model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= rf_best_threshold).astype(int)
            rf_f1 = f1_score(y_val, y_pred)
            rf_precision = precision_score(y_val, y_pred)
            rf_recall = recall_score(y_val, y_pred)
            
            print("\nRandom Forest 詳細驗證結果:")
            print(f"最佳參數: {rf_best_params}")
            print(f"最佳閾值: {rf_best_threshold:.4f}")
            print(f"驗證集 F1 分數: {rf_f1:.4f}")
            print(f"驗證集 Precision: {rf_precision:.4f}")
            print(f"驗證集 Recall: {rf_recall:.4f}")
            
            # 檢查預測分布
            positives = np.sum(y_pred)
            total = len(y_pred)
            print(f"預測正例數: {positives}/{total} ({positives/total:.2%})")
            print(f"實際正例數: {np.sum(y_val)}/{len(y_val)} ({np.sum(y_val)/len(y_val):.2%})")
        else:
            print("\nRandom Forest 模型訓練失敗")
            rf_f1 = 0
        
        # 合併所有結果
        all_summaries = []
        all_summaries.extend(train_summaries)
        all_summaries.extend(val_summaries)
        all_summaries.extend(test_summaries)
        
        # 保存所有結果
        save_combined_results(all_summaries, param_search_results, output_path)
        
        # 保存詳細的模型比較結果
        detailed_comparison = save_detailed_model_comparison(train_results, val_results, test_results, output_path)
        
        # 準備模型資訊 (僅 XGBoost 和 LightGBM)
        xgb_info = {
            'model': xgb_best_model,
            'threshold': xgb_best_threshold,
            'params': xgb_best_params,
            'f1_score': xgb_f1 if 'xgb_f1' in locals() else 0
        }
        
        lgb_info = {
            'model': lgb_best_model,
            'threshold': lgb_best_threshold,
            'params': lgb_best_params,
            'f1_score': lgb_f1 if 'lgb_f1' in locals() else 0
        }
        
        # 新增 Random Forest 資訊
        rf_info = {
            'model': rf_best_model,
            'threshold': rf_best_threshold,
            'params': rf_best_params,
            'f1_score': rf_f1 if 'rf_f1' in locals() else 0
        }
        
        # 保存模型資訊到文字檔 (僅 XGBoost 和 LightGBM)
        save_model_info(output_path, xgb_info, lgb_info, rf_info)
        
        # 保存模型 (僅 XGBoost 和 LightGBM)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存 XGBoost 模型
        xgb_model_path = os.path.join(output_path, f'xgboost_best_model_{timestamp}.pkl')
        with open(xgb_model_path, 'wb') as f:
            pickle.dump(xgb_info, f)
        print(f"XGBoost 最佳模型已保存至: {xgb_model_path}")
        
        # 保存 LightGBM 模型
        lgb_model_path = os.path.join(output_path, f'lightgbm_best_model_{timestamp}.pkl')
        with open(lgb_model_path, 'wb') as f:
            pickle.dump(lgb_info, f)
        print(f"LightGBM 最佳模型已保存至: {lgb_model_path}")
        
        # 保存 Random Forest 模型
        rf_model_path = os.path.join(output_path, f'random_forest_best_model_{timestamp}.pkl')
        with open(rf_model_path, 'wb') as f:
            pickle.dump(rf_info, f)
        print(f"Random Forest 最佳模型已保存至: {rf_model_path}")
        
        # 顯示總結
        print("\n=== 程式執行總結 ===")
        print(f"總共訓練和評估了5種模型: XGBoost, LightGBM, Random Forest, SVM, 神經網絡")
        print(f"所有模型評估結果皆已保存至CSV和Excel檔案")
        print(f"按照要求，只有XGBoost和LightGBM模型被保存為可用模型")
        print("\n8. 程式執行完成")
        
    except Exception as e:
        print(f"執行過程中發生錯誤: {str(e)}")
        traceback.print_exc()

def save_combined_results(model_results, param_search_results, output_path):
    """
    將模型評估結果和參數搜尋結果合併到CSV檔案和Excel檔案中
    """
    # 建立一個空的 DataFrame 來存儲所有結果
    all_results = pd.DataFrame()
    
    # 處理模型評估結果
    evaluation_df = pd.DataFrame(model_results)
    
    # 處理參數搜尋結果
    param_search_dfs = []
    for model_name, params_list in param_search_results.items():
        if params_list:  # 確保有參數搜尋結果
            df = pd.DataFrame(params_list)
            df['模型'] = model_name
            param_search_dfs.append(df)
    
    if param_search_dfs:
        param_search_combined = pd.concat(param_search_dfs, ignore_index=True)
        
        # 將參數搜尋結果中的數值格式化為4位小數
        numeric_columns = param_search_combined.select_dtypes(include=['float64', 'float32']).columns
        for col in numeric_columns:
            param_search_combined[col] = param_search_combined[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else None)
    
    # 保存到 Excel 檔案的不同工作表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = f"{output_path}/model_results_{timestamp}.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        evaluation_df.to_excel(writer, sheet_name='模型評估結果', index=False)
        if param_search_dfs:
            param_search_combined.to_excel(writer, sheet_name='參數搜尋結果', index=False)
    
    # 保存CSV版本的評估結果 - 所有模型的結果都會包含在這裡
    csv_path = f"{output_path}/model_evaluation_{timestamp}.csv"
    evaluation_df.to_csv(csv_path, index=False)
    
    # 針對每個模型單獨保存詳細的參數搜尋結果CSV
    for model_name, params_list in param_search_results.items():
        if params_list:
            model_params_df = pd.DataFrame(params_list)
            model_csv_path = f"{output_path}/{model_name}_parameters_{timestamp}.csv"
            model_params_df.to_csv(model_csv_path, index=False)
            print(f"{model_name}參數搜尋結果已保存至: {model_csv_path}")
    
    print(f"\n所有模型評估結果已保存至: {excel_path}")
    print(f"評估結果CSV已保存至: {csv_path}")
    print("\n=== 模型評估結果摘要 ===")
    print(evaluation_df.to_string(index=False))
    
    if param_search_dfs:
        print("\n=== 參數搜尋結果摘要 (各前3筆) ===")
        for model_name, params_list in param_search_results.items():
            if params_list:
                top_params = pd.DataFrame(params_list).sort_values(by='F1分數', ascending=False).head(3)
                print(f"\n{model_name}最佳參數組合:")
                print(top_params[['F1分數', '最佳閾值', '參數組合']].to_string(index=False))

# 新增函數：將所有模型的詳細評估結果保存成一個易讀的CSV文件
def save_detailed_model_comparison(train_results, val_results, test_results, output_path):
    """將所有模型在訓練、驗證和測試集上的評估結果整合成一個詳細比較表"""
    # 創建一個包含所有結果的DataFrame
    comparison_df = pd.DataFrame(columns=[
        '模型', '數據集', '準確率', '精確率', '召回率', 'F1分數', 'AUC', '最佳閾值'
    ])
    
    # 整合所有結果
    row_index = 0
    for dataset_name, results in [('訓練集', train_results), ('驗證集', val_results), ('測試集', test_results)]:
        for result in results:
            comparison_df.loc[row_index] = [
                result['model_name'],
                dataset_name,
                result['accuracy'],
                result['precision'],
                result['recall'],
                result['f1'],
                result['auc'],
                result['threshold']
            ]
            row_index += 1
    
    # 保存結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_csv_path = f"{output_path}/detailed_model_comparison_{timestamp}.csv"
    comparison_df.to_csv(detailed_csv_path, index=False)
    
    print(f"\n詳細模型比較結果已保存至: {detailed_csv_path}")
    
    return comparison_df

def save_model_info(output_path, xgb_info, lgb_info, rf_info):  # 添加 rf_info 參數
    """保存模型資訊到文字檔"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型資訊
    model_info_path = os.path.join(output_path, f'best_model_info_{timestamp}.txt')
    with open(model_info_path, 'w', encoding='utf-8') as f:
        f.write("=== XGBoost 模型資訊 ===\n")
        if xgb_info['model'] is not None:
            f.write(f"最佳參數: {xgb_info['params']}\n")
            f.write(f"驗證集 F1 分數: {xgb_info['f1_score']:.4f}\n")
            f.write(f"最佳閾值: {xgb_info['threshold']:.4f}\n\n")
        else:
            f.write("XGBoost 模型訓練失敗\n\n")
            
        f.write("=== LightGBM 模型資訊 ===\n")
        if lgb_info['model'] is not None:
            f.write(f"最佳參數: {lgb_info['params']}\n")
            f.write(f"驗證集 F1 分數: {lgb_info['f1_score']:.4f}\n")
            f.write(f"最佳閾值: {lgb_info['threshold']:.4f}\n\n")
        else:
            f.write("LightGBM 模型訓練失敗\n\n")

        # 新增 Random Forest 資訊
        f.write("=== Random Forest 模型資訊 ===\n")
        if rf_info['model'] is not None:
            f.write(f"最佳參數: {rf_info['params']}\n")
            f.write(f"驗證集 F1 分數: {rf_info['f1_score']:.4f}\n")
            f.write(f"最佳閾值: {rf_info['threshold']:.4f}\n\n")
        else:
            f.write("Random Forest 模型訓練失敗\n\n")
            
        # 寫入整體最佳模型資訊
        f.write("=== 整體最佳模型 ===\n")
        best_model = max([
            ('XGBoost', xgb_info['f1_score']),
            ('LightGBM', lgb_info['f1_score']),
            ('Random Forest', rf_info['f1_score'])
        ], key=lambda x: x[1])
        f.write(f"最佳模型: {best_model[0]}\n")
        f.write(f"最佳 F1 分數: {best_model[1]:.4f}\n")
    
    # 保存閾值資訊
    threshold_info_path = os.path.join(output_path, f'best_threshold_info_{timestamp}.txt')
    with open(threshold_info_path, 'w', encoding='utf-8') as f:
        f.write("=== XGBoost 閾值資訊 ===\n")
        if xgb_info['model'] is not None:
            f.write(f"最佳閾值: {xgb_info['threshold']:.4f}\n")
            f.write(f"驗證集 F1 分數: {xgb_info['f1_score']:.4f}\n\n")
        else:
            f.write("XGBoost 模型訓練失敗\n\n")
            
        f.write("=== LightGBM 閾值資訊 ===\n")
        if lgb_info['model'] is not None:
            f.write(f"最佳閾值: {lgb_info['threshold']:.4f}\n")
            f.write(f"驗證集 F1 分數: {lgb_info['f1_score']:.4f}\n\n")
        else:
            f.write("LightGBM 模型訓練失敗\n\n")

        # 新增 Random Forest 閾值資訊
        f.write("=== Random Forest 閾值資訊 ===\n")
        if rf_info['model'] is not None:
            f.write(f"最佳閾值: {rf_info['threshold']:.4f}\n")
            f.write(f"驗證集 F1 分數: {rf_info['f1_score']:.4f}\n\n")
        else:
            f.write("Random Forest 模型訓練失敗\n\n")
            
        # 寫入整體最佳模型的閾值
        f.write("=== 整體最佳模型閾值 ===\n")
        if best_model[0] == 'XGBoost':
            f.write(f"最佳閾值: {xgb_info['threshold']:.4f}\n")
        elif best_model[0] == 'LightGBM':
            f.write(f"最佳閾值: {lgb_info['threshold']:.4f}\n")
        else:  # Random Forest
            f.write(f"最佳閾值: {rf_info['threshold']:.4f}\n")

# 添加新的合併比較圖表函數
def plot_combined_classification_comparison(train_results, val_results, test_results, output_path):
    """繪製三組資料的分類評估指標比較圖"""
    plt.figure(figsize=(15, 8))
    
    # 獲取所有模型名稱
    models = [result['model_name'] for result in train_results]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    # 設定繪圖參數
    x = np.arange(len(models))
    width = 0.25  # 柱狀圖寬度
    
    # 繪製每個指標的柱狀圖
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        # 獲取三組數據
        train_scores = [result[metric.lower()] for result in train_results]
        val_scores = [result[metric.lower()] for result in val_results]
        test_scores = [result[metric.lower()] for result in test_results]
        
        # 繪製三組柱狀圖
        plt.bar(x - width, train_scores, width, label='Training', color='skyblue')
        plt.bar(x, val_scores, width, label='Validation', color='lightgreen')
        plt.bar(x + width, test_scores, width, label='Testing', color='salmon')
        
        plt.xlabel('Model')
        plt.ylabel(f'{metric} Score')
        plt.title(f'{metric} Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 修改保存路徑，使用傳入的 output_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'{output_path}/combined_metrics_comparison_{timestamp}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

# 添加一個視覺化函數，生成各模型在不同數據集上的F1分數比較圖
def plot_f1_comparison(train_results, val_results, test_results, output_path):
    """生成一個比較所有模型在三個數據集上F1分數的柱狀圖"""
    plt.figure(figsize=(12, 8))
    
    # 準備數據
    models = [result['model_name'] for result in train_results]
    train_f1 = [result['f1'] for result in train_results]
    val_f1 = [result['f1'] for result in val_results]
    test_f1 = [result['f1'] for result in test_results]
    
    # 設置柱狀圖參數
    x = np.arange(len(models))
    width = 0.25
    
    # 繪製柱狀圖
    plt.bar(x - width, train_f1, width, label='Training', color='skyblue')
    plt.bar(x, val_f1, width, label='Validation', color='lightgreen')
    plt.bar(x + width, test_f1, width, label='Testing', color='salmon')
    
    # 添加標籤和圖例 
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.title('F1 Score Comparison Across Datasets', fontsize=16)
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 在柱狀圖上添加數值標籤
    for i, v in enumerate(train_f1):
        plt.text(i - width, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(val_f1):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(test_f1):
        plt.text(i + width, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存圖片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'{output_path}/f1_score_comparison_{timestamp}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"F1分數比較圖已保存至: {output_path}/f1_score_comparison_{timestamp}.png")

# 繪製組合ROC曲線比較圖
def plot_combined_roc_curves(train_results, val_results, test_results, 
                           X_train, y_train, X_val, y_val, X_test, y_test, 
                           models, X_train_scaled, X_val_scaled, X_test_scaled,
                           output_path):
    """Plot ROC curves for all three datasets in a single figure"""
    plt.switch_backend('Agg')
    plt.figure(figsize=(18, 6))
    
    # 分別繪製訓練、驗證和測試集的ROC曲線
    datasets = [
        ('Training Set', X_train, y_train, X_train_scaled, train_results),
        ('Validation Set', X_val, y_val, X_val_scaled, val_results),
        ('Testing Set', X_test, y_test, X_test_scaled, test_results)
    ]
    
    for i, (dataset_name, X, y, X_scaled, results) in enumerate(datasets):
        plt.subplot(1, 3, i+1)
        
        for result in results:
            model_name = result['model_name']
            model = models[model_name]
            
            # 針對SVM和神經網絡使用標準化的數據
            if model_name in ['SVM', 'Neural Network']:
                X_data = X_scaled
            else:
                X_data = X
            
            # 獲取預測概率
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_data)[:, 1]
            else:
                y_prob = model.decision_function(X_data)

            # 計算ROC曲線
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc = auc(fpr, tpr)

            # 繪製ROC曲線
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')

        # 繪製隨機猜測的基準線
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{dataset_name} ROC Curves', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('ROC Curve Comparison Across All Datasets', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存圖片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'{output_path}/combined_roc_curves_{timestamp}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Combined ROC curves saved to: {save_path}")
    return save_path

if __name__ == "__main__":
    main()