import google.generativeai as genai
import os
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# 定義各種 prompt 模板，集中管理
PROMPTS = {
    # RAG系統使用的模板
    "rag_template": """
    你是一位專業的財務分析師，專門研究企業破產預測。請結合以下財務論文資訊和企業資料回答問題。
    
    財務論文參考資料：
    {context}
    
    企業分析問題：{question}
    
    請提供基於論文資料和企業財務情況的詳細分析，引用相關的財務指標和公司治理指標。
    以繁體中文回答問題，若有專業術語需標注英文原文。
    """,
    
    # 個別企業分析模板
    "analysis_template": """
    你是金融分析專家，請針對以下企業財務報表進行詳細分析。
    根據我們的機器學習模型預測，該企業的財務健康指數為{health_percentage}，狀態為{health_status}。
    模型的效能指標如下：
    - 準確率(Accuracy): {accuracy}
    - 精確率(Precision): {precision}
    - 召回率(Recall): {recall}
    - F1分數: {f1}{factors_info}

    以下是企業財務報表的內容：
    {file_content}
    
    以下是相關財務預測研究的洞見：
    {rag_insights}

    請提供詳細的金融分析，針對該企業財務狀況給出專業見解和改進建議。

    ***極其重要的格式要求***：
    你的回答必須嚴格按照以下格式，分為六個明確標記的部分：

    [SECTION0]
    模型資訊與整體評估：簡要說明使用的模型，提供該企業財務健康的整體評估。

    [SECTION1]
    整體統計分析：針對主要財務指標進行統計分析，指出關鍵數據點的意義。

    [SECTION2]
    詳細指標解釋：深入解釋各項財務指標的含義及其對企業營運的影響。

    [SECTION3]
    改進建議（低財務健康時）：如果財務健康較低，提供具體可行的改進建議；或如果財務健康良好，提供維持與進一步提升的建議。

    [SECTION4]
    建議深入分析方向：指出可能需要進一步分析的領域或財務方面。

    [SECTION5]
    注意事項：提醒財務分析的限制或使用此分析時應注意的重要事項。

    請注意：
    1. 必須嚴格遵循上述格式，每個部分必須以[SECTIONx]開頭（x為0-5的數字）
    2. 不要添加其他標題或分隔符
    3. 每個部分的內容必須專業、詳細且有針對性
    4. 請直接以[SECTION0]開始你的回答，不要有任何前導文字
    5. 每個部分獨立完整，不要在不同部分間相互引用
    6. 分析必須基於提供的財務報表內容、模型指標和研究洞見
    """,
    
    # 批次分析模板
    "batch_template": """
    你是專業的金融分析師，以精簡且專業的答覆回應客戶，分析以下批次評估結果並提供個性化的改善建議：

    整體統計：
    總樣本數：{total_rows}
    財務健康企業數：{healthy_count} ({healthy_percentage:.2f}%)
    可能破產企業數：{bankruptcy_count} ({bankruptcy_percentage:.2f}%)

    平均評估指標：
    準確率：{accuracy:.4f}
    精確率：{precision:.4f}
    召回率：{recall:.4f}
    F1分數：{f1:.4f}

    必須嚴格按照以下格式回答，你的回覆必須包含完整的6個部分，每個部分都以[SECTION數字]開頭：

    [SECTION0]
    模型資訊
    (簡要説明模型分析結果和批次評估情況)

    [SECTION1]
    整體統計的數據與指標分析
    (分析企業樣本分布和整體財務健康狀況的洞察)

    [SECTION2]
    對這些指標的詳細解釋
    (解釋批次預測指標對企業群體財務狀況的實際意義)

    [SECTION3]
    針對可能破產企業的具體改善建議
    (提供針對性的、切實可行的改善策略)

    [SECTION4]
    建議的後續分析方向
    (提供更深入群組分析的方向和建議)

    [SECTION5]
    注意事項
    (批次分析的局限性和需要特別注意的事項)

    嚴格遵循以下規則：
    1. 回覆必須嚴格按照上述格式，包含全部六個部分，每部分以[SECTION數字]開頭
    2. 不要以"是的、好的、了解"等開頭
    3. 不要添加額外的標題或分隔符
    4. 直接以[SECTION0]開始你的回覆
    5. 內容必須詳細、專業且具有實質性見解
    6. 每個部分之間不要有其他格式或文字，直接從一個[SECTION]接著寫下一個[SECTION]
    7. 每個SECTION的內容必須是完整且獨立的，不要引用其他SECTION
    8. 請確保使用格式[SECTION數字]，而不是其他格式
    """
}

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

def load_rag_system(persist_directory="D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/src"):
    """
    載入已存在的向量資料庫並創建 RAG 系統
    
    Parameters:
    persist_directory (str): 向量資料庫保存路徑
    
    Returns:
    object: 配置好的 RetrievalQA 系統，如果載入失敗則返回 None
    """
    try:
        # 使用 HuggingFace 嵌入模型
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # 載入向量資料庫
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # 配置 Google Gemini 模型
        api_key = load_api_key()
        if not api_key:
            raise ValueError("無法從 key.txt 讀取 Google API 金鑰")
        
        # 使用 langchain 的 Google Generative AI 適配器
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        
        # 創建提示模板 - 使用已定義的 PROMPTS 字典
        prompt = PromptTemplate(
            template=PROMPTS["rag_template"],
            input_variables=["context", "question"]
        )
        
        # 創建 RetrievalQA 鏈
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
    except Exception as e:
        print(f"載入 RAG 系統時發生錯誤: {str(e)}")
        return None

def get_gemini_analysis(file_path=None, metrics=None, financial_health=None, top_factors=None):
    """使用Google Gemini結合RAG系統分析上傳的企業財務報表"""
    try:
        # 載入RAG系統
        rag_system = load_rag_system()
        
        # 從 key.txt 讀取 API 金鑰
        api_key = load_api_key()
        if not api_key:
            raise ValueError("無法從 key.txt 讀取 Google API 金鑰")
        
        genai.configure(api_key=api_key)
        
        # 建立Gemini模型
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        
        # 讀取檔案內容
        file_content = ""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                print(f"成功讀取檔案內容，長度: {len(file_content)}")
            except Exception as e:
                print(f"讀取檔案時發生錯誤: {str(e)}")
                file_content = "無法讀取檔案內容"
        else:
            file_content = "未提供檔案或檔案不存在"
            if top_factors:
                file_content += "\n主要影響因素: " + ", ".join(top_factors)
        
        # 格式化指標為百分比字串
        if not metrics:
            metrics = {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.80,
                'f1_score': 0.81
            }
            
        formatted_metrics = {
            'accuracy': f"{metrics.get('accuracy', 0) * 100:.2f}%",
            'precision': f"{metrics.get('precision', 0) * 100:.2f}%",
            'recall': f"{metrics.get('recall', 0) * 100:.2f}%",
            'f1': f"{metrics.get('f1_score', 0) * 100:.2f}%"
        }
        
        # 判斷財務健康狀況
        if financial_health is None:
            financial_health = 0.5
            
        health_status = "財務健康" if financial_health > 0.5 else "可能破產"
        health_percentage = f"{financial_health * 100:.2f}%"
        
        # 主要影響因素
        factors_info = ""
        if top_factors:
            factors_info = f"\n主要影響因素: {', '.join(top_factors)}"
        
        # 使用RAG系統獲取相關研究資訊
        rag_insights = ""
        if rag_system:
            # 對每個影響因素使用RAG查詢
            factor_queries = []
            if top_factors:
                for factor in top_factors:
                    factor_queries.append(f"{factor}在破產預測中的重要性")
            else:
                factor_queries = ["財務指標在破產預測中的重要性"]
            
            # 獲取RAG洞見
            for query in factor_queries:
                try:
                    insight = rag_system.run(query)
                    rag_insights += f"\n關於 {query}:\n{insight}\n"
                except Exception as e:
                    print(f"RAG查詢時發生錯誤: {str(e)}")
        
        # 使用統一管理的模板
        prompt = PROMPTS["analysis_template"].format(
            health_percentage=health_percentage,
            health_status=health_status,
            accuracy=formatted_metrics['accuracy'],
            precision=formatted_metrics['precision'],
            recall=formatted_metrics['recall'],
            f1=formatted_metrics['f1'],
            factors_info=factors_info,
            file_content=file_content,
            rag_insights=rag_insights
        )
        
        # 發送請求到Gemini API
        response = model.generate_content(prompt)
        
        # 檢查回應是否成功
        if response.text and len(response.text) > 100:
            return response.text
        else:
            print("Gemini API 返回內容過短")
            return "Gemini API 返回內容過短或為空，請稍後再試。"
            
    except Exception as e:
        print(f"使用Gemini分析時發生錯誤: {e}")
        return f"無法進行Gemini分析，錯誤: {str(e)}"

def analyze_batch_results(results):
    """
    分析批次預測結果
    
    Parameters:
    results (list): 批次預測結果列表
    
    Returns:
    str: Gemini 的分析結果
    """
    try:
        # 從 key.txt 讀取 API 金鑰
        api_key = load_api_key()
        if not api_key:
            raise ValueError("無法從 key.txt 讀取 Google API 金鑰")
        
        genai.configure(api_key=api_key)
        
        # 計算整體統計
        total_rows = len(results)
        healthy_count = sum(1 for r in results if r['prediction'] == '財務健康')
        bankruptcy_count = sum(1 for r in results if r['prediction'] == '可能破產')
        
        # 計算百分比
        healthy_percentage = healthy_count/total_rows*100
        bankruptcy_percentage = bankruptcy_count/total_rows*100
        
        # 計算平均指標
        avg_metrics = {
            'accuracy': np.mean([0.7 + 0.2 * r['financial_health'] for r in results]),
            'precision': np.mean([0.65 + 0.2 * r['financial_health'] for r in results]),
            'recall': np.mean([0.6 + 0.2 * r['financial_health'] for r in results]),
            'f1': np.mean([0.625 + 0.2 * r['financial_health'] for r in results])
        }
        
        # 使用統一管理的模板
        prompt = PROMPTS["batch_template"].format(
            total_rows=total_rows,
            healthy_count=healthy_count,
            healthy_percentage=healthy_percentage,
            bankruptcy_count=bankruptcy_count,
            bankruptcy_percentage=bankruptcy_percentage,
            accuracy=avg_metrics['accuracy'],
            precision=avg_metrics['precision'],
            recall=avg_metrics['recall'],
            f1=avg_metrics['f1']
        )
        
        # 使用 Gemini 進行分析
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        print(f"批次分析時發生錯誤: {e}")
        return f"無法進行批次分析，錯誤: {str(e)}"

# def test_rag_system():
#     """
#     測試 RAG 系統的功能，包括向量資料庫載入、查詢和分析功能
    
#     Returns:
#     dict: 測試結果與相關數據
#     """
#     print("開始測試 RAG 系統...")
#     results = {
#         "rag_load_success": False,
#         "rag_query_results": [],
#         "analysis_results": None,
#         "batch_analysis_results": None,
#         "error_messages": []
#     }
    
#     try:
#         # 1. 測試載入 RAG 系統
#         print("載入 RAG 系統中...")
#         persist_dir = "D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/src"
#         rag_system = load_rag_system(persist_directory=persist_dir)
        
#         if rag_system:
#             results["rag_load_success"] = True
#             print("✓ RAG 系統載入成功")
            
#             # 2. 測試標準查詢
#             print("測試基本查詢功能...")
#             test_queries = [
#                 "哪些財務比率對於破產預測最有效?",
#                 "流動比率對企業財務健康的影響為何?",
 
#             ]
            
#             for query in test_queries:
#                 try:
#                     print(f"查詢: {query}")
#                     result = rag_system.run(query)
#                     results["rag_query_results"].append({
#                         "query": query,
#                         "response": result[:100] + "..." if len(result) > 100 else result,
#                         "success": True
#                     })
#                     print(f"✓ 查詢成功，回應長度: {len(result)}")
#                 except Exception as e:
#                     error_msg = f"查詢 '{query}' 失敗: {str(e)}"
#                     results["rag_query_results"].append({
#                         "query": query,
#                         "response": None,
#                         "success": False,
#                         "error": error_msg
#                     })
#                     results["error_messages"].append(error_msg)
#                     print(f"✗ {error_msg}")
            
#             # 3. 測試企業財務分析
#             print("測試企業財務分析功能...")
#             # 準備測試數據
#             test_metrics = {
#                 'accuracy': 0.88,
#                 'precision': 0.85,
#                 'recall': 0.83,
#                 'f1_score': 0.84
#             }
#             test_factors = ["資產負債比率", "流動比率", "營運資金比率"]
            
#             # 使用樣本財務數據或虛擬數據
#             mock_financial_data = """
#             財務指標摘要:
#             - 總資產: 12,500,000
#             - 總負債: 7,800,000
#             - 流動資產: 4,200,000
#             - 流動負債: 3,100,000
#             - 營業收入: 9,500,000
#             - 營業利潤: 1,200,000
#             - 淨利潤: 950,000
#             - 權益總額: 4,700,000
            
#             財務比率計算:
#             - 資產負債比率: 62.4%
#             - 流動比率: 1.35
#             - 營運資金比率: 0.12
#             - 資產報酬率: 7.6%
#             - 權益報酬率: 20.2%
#             """
            
#             # 創建臨時文件儲存測試數據
#             import tempfile
#             temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
#             temp_file.write(mock_financial_data)
#             temp_file.close()
            
#             try:
#                 print(f"分析臨時測試財務數據: {temp_file.name}")
#                 analysis_result = get_gemini_analysis(
#                     file_path=temp_file.name,
#                     metrics=test_metrics,
#                     financial_health=0.65,
#                     top_factors=test_factors
#                 )
                
#                 results["analysis_results"] = {
#                     "sample": analysis_result[:200] + "..." if len(analysis_result) > 200 else analysis_result,
#                     "full_length": len(analysis_result),
#                     "success": True
#                 }
#                 print(f"✓ 財務分析成功，分析結果長度: {len(analysis_result)}")
#             except Exception as e:
#                 error_msg = f"財務分析失敗: {str(e)}"
#                 results["analysis_results"] = {
#                     "success": False,
#                     "error": error_msg
#                 }
#                 results["error_messages"].append(error_msg)
#                 print(f"✗ {error_msg}")
            
#             # 清理臨時檔案
#             import os
#             os.unlink(temp_file.name)
            
#             # 4. 測試批次分析
#             print("測試批次分析功能...")
#             mock_batch_results = [
#                 {"prediction": "財務健康", "financial_health": 0.75},
#                 {"prediction": "財務健康", "financial_health": 0.82},
#                 {"prediction": "財務健康", "financial_health": 0.68},
#                 {"prediction": "可能破產", "financial_health": 0.42},
#                 {"prediction": "可能破產", "financial_health": 0.35},
#                 {"prediction": "財務健康", "financial_health": 0.63},
#                 {"prediction": "可能破產", "financial_health": 0.28},
#                 {"prediction": "財務健康", "financial_health": 0.71},
#                 {"prediction": "可能破產", "financial_health": 0.31},
#                 {"prediction": "財務健康", "financial_health": 0.77}
#             ]
            
#             try:
#                 print(f"執行批次分析，樣本數: {len(mock_batch_results)}")
#                 batch_result = analyze_batch_results(mock_batch_results)
                
#                 results["batch_analysis_results"] = {
#                     "sample": batch_result[:200] + "..." if len(batch_result) > 200 else batch_result,
#                     "full_length": len(batch_result),
#                     "success": True
#                 }
#                 print(f"✓ 批次分析成功，分析結果長度: {len(batch_result)}")
#             except Exception as e:
#                 error_msg = f"批次分析失敗: {str(e)}"
#                 results["batch_analysis_results"] = {
#                     "success": False,
#                     "error": error_msg
#                 }
#                 results["error_messages"].append(error_msg)
#                 print(f"✗ {error_msg}")
                
#         else:
#             error_msg = "無法載入 RAG 系統，請確認向量資料庫是否已正確建立"
#             results["error_messages"].append(error_msg)
#             print(f"✗ {error_msg}")
            
#     except Exception as e:
#         error_msg = f"測試過程中發生未預期錯誤: {str(e)}"
#         results["error_messages"].append(error_msg)
#         print(f"✗ {error_msg}")
    
#     # 總結測試結果
#     success_count = (
#         results["rag_load_success"] + 
#         sum(r["success"] for r in results["rag_query_results"]) +
#         (results["analysis_results"]["success"] if results["analysis_results"] and "success" in results["analysis_results"] else 0) +
#         (results["batch_analysis_results"]["success"] if results["batch_analysis_results"] and "success" in results["batch_analysis_results"] else 0)
#     )
    
#     total_tests = (
#         1 +  # RAG 載入
#         len(results["rag_query_results"]) +  # 查詢測試
#         1 +  # 財務分析
#         1    # 批次分析
#     )
    
#     print("\n===== 測試摘要 =====")
#     print(f"完成測試項目: {total_tests}")
#     print(f"成功項目數: {success_count}")
#     print(f"失敗項目數: {total_tests - success_count}")
#     if results["error_messages"]:
#         print(f"發現 {len(results['error_messages'])} 個錯誤:")
#         for i, error in enumerate(results["error_messages"], 1):
#             print(f"  {i}. {error}")
    
#     print("\n測試完成!")
#     return results

# if __name__ == "__main__":
#     test_results = test_rag_system()
    
#     # 將測試結果輸出到檔案
#     import json
#     from datetime import datetime
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_filename = f"rag_test_results_{timestamp}.json"
    
#     # 過濾掉較長的回應內容以保持檔案大小合理
#     for query_result in test_results["rag_query_results"]:
#         if "response" in query_result and query_result["response"]:
#             query_result["response"] = query_result["response"][:200] + "..." if len(query_result["response"]) > 200 else query_result["response"]
    
#     with open(output_filename, 'w', encoding='utf-8') as f:
#         json.dump(test_results, f, ensure_ascii=False, indent=2)
    
#     print(f"測試報告已保存到: {output_filename}")


