from gemini_analysis import get_gemini_analysis, analyze_batch_results
import csv

def test_get_gemini_analysis():
    # 準備測試數據
    top_factors = []  # 移除不需要的因子
    financial_health = 0.85
    metrics = {
        'accuracy': 0.9,  # 假設需要此指標
        'precision': 0.85,  # 假設需要此指標
        'recall': 0.8,  # 假設需要此指標
        'f1': 0.82  # 假設需要此指標
    }
    
    # 調用函數
    try:
        result = get_gemini_analysis(top_factors, financial_health, metrics)
        print("get_gemini_analysis result:", result)
    except Exception as e:
        print("Error in get_gemini_analysis:", e)

def test_analyze_batch_results():
    # 準備測試數據
    results = []
    with open("D:/Shawn/python_class/Tibame/團專/data/test_data0313.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # 確保 'prediction' 和 'financial_health' 鍵存在
            if 'prediction' not in row:
                row['prediction'] = 'Unknown'  # 或者根據需要設置預設值
            if 'financial_health' not in row:
                row['financial_health'] = 0.5  # 設置一個合理的預設值
            results.append(row)
    
    # 調用函數
    try:
        analysis = analyze_batch_results(results)
        print("analyze_batch_results result:", analysis)
    except Exception as e:
        print("Error in analyze_batch_results:", e)

if __name__ == "__main__":
    # test_get_gemini_analysis()
    test_analyze_batch_results()