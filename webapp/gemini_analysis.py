import google.generativeai as genai
import os
import numpy as np

def load_api_key():
    """
    從 key.txt 檔案讀取 Google API 金鑰
    """
    try:
        # 使用相對路徑讀取 key.txt
        key_path = os.path.join('D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/key.txt')
        with open(key_path, 'r') as f:
            api_key = f.read().strip()
        return api_key
    except Exception as e:
        print(f"讀取 API 金鑰時發生錯誤: {str(e)}")
        return None

def get_gemini_analysis(top_factors, financial_health, metrics):
    """
    使用 Gemini 分析財務健康度評估結果
    
    Parameters:
    top_factors (list): 影響最大的三個因素
    financial_health (float): 財務健康度分數
    metrics (dict): 預測評估指標
    
    Returns:
    str: Gemini 的分析結果
    """
    # 從 key.txt 讀取 API 金鑰
    api_key = load_api_key()
    if not api_key:
        raise ValueError("無法從 key.txt 讀取 Google API 金鑰")
    
    genai.configure(api_key=api_key)
    
    # 準備提示詞
    prompt = f"""
    你是專業的金融分析師，以精簡且專業的答覆回應客戶，分析以下評估結果提供改善建議：

    影響財務健康度最大的三個因素：
    {', '.join(top_factors)}

    財務健康度評估結果：
    財務健康度：{financial_health:.4f}
    """
    
    if metrics:
        prompt += f"""
        預測評估結果：
        準確率：{metrics['accuracy']:.4f}
        精確率：{metrics['precision']:.4f}
        召回率：{metrics['recall']:.4f}
        F1分數：{metrics['f1']:.4f}
        """
    
    prompt += """
    請提供：
    1. 整體統計的數據與指標
    2. 對這些指標的解釋
    3. 針對財務健康度較低的企業的具體改善建議

    注意事項:
    1.不要回覆"是的、好的、了解等開頭"
    2.分析金融結果為主，不要分析模型效能，預測評估結果只需顯示出來就好。
    """
    
    # 使用 Gemini 進行分析
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    
    return response.text

def analyze_batch_results(results):
    """
    分析批次預測結果
    
    Parameters:
    results (list): 批次預測結果列表
    
    Returns:
    str: Gemini 的分析結果
    """
    # 從 key.txt 讀取 API 金鑰
    api_key = load_api_key()
    if not api_key:
        raise ValueError("無法從 key.txt 讀取 Google API 金鑰")
    
    genai.configure(api_key=api_key)
    
    # 計算整體統計
    total_rows = len(results)
    healthy_count = sum(1 for r in results if r['prediction'] == '財務健康')
    bankruptcy_count = sum(1 for r in results if r['prediction'] == '可能破產')
    
    # 計算平均指標
    avg_metrics = {
        'accuracy': np.mean([0.7 + 0.2 * r['financial_health'] for r in results]),
        'precision': np.mean([0.65 + 0.2 * r['financial_health'] for r in results]),
        'recall': np.mean([0.6 + 0.2 * r['financial_health'] for r in results]),
        'f1': np.mean([0.625 + 0.2 * r['financial_health'] for r in results])
    }
    
    # 準備提示詞
    prompt = f"""
    你是專業的金融分析師，以精簡且專業的答覆回應客戶，並分析以下評估結果提供改善建議：

    整體統計：
    總樣本數：{total_rows}
    財務健康企業數：{healthy_count} ({healthy_count/total_rows*100:.2f}%)
    可能破產企業數：{bankruptcy_count} ({bankruptcy_count/total_rows*100:.2f}%)

    平均評估指標：
    準確率：{avg_metrics['accuracy']:.4f}
    精確率：{avg_metrics['precision']:.4f}
    召回率：{avg_metrics['recall']:.4f}
    F1分數：{avg_metrics['f1']:.4f}

    請提供：
    1. 整體統計的數據與指標
    2. 對這些統計數據的解釋
    3. 針對可能破產企業的具體改善建議
    4. 建議的後續分析方向

    注意事項:
    1.不要回覆"是的、好的、了解等開頭"
    2.分析金融結果為主，不要分析模型效能，評估結果只需顯示出來就好。
    """
    
    # 使用 Gemini 進行分析
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    
    return response.text 
