import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

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

def create_RAG_system(persist_directory=None):
    """
    創建 RAG 系統，用於財務破產預測分析
    
    Parameters:
    persist_directory (str): 向量資料庫保存路徑，預設為None
    
    Returns:
    object: 配置好的 RetrievalQA 系統
    """
    # 設定文件路徑
    pdf_path = "D:/Shawn/python_class/Tibame/團專/Financial ratios and corporate governance indicators in bankruptcy prediction A comprehensive study.pdf"
    
    # 如果向量資料庫目錄不存在，創建該目錄
    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
    
    # 載入並分割文件
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 對文件進行分塊
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # 使用 HuggingFace 嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # 創建向量存儲並保存到指定目錄
    if persist_directory:
        db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
        db.persist()  # 保存向量資料庫
        print(f"向量資料庫已保存到: {persist_directory}")
    else:
        db = Chroma.from_documents(chunks, embeddings)
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # 配置 Google Gemini 模型
    api_key = load_api_key()
    if not api_key:
        raise ValueError("無法從 key.txt 讀取 Google API 金鑰")
    
    genai.configure(api_key=api_key)
    
    # 使用 langchain 的 Google Generative AI 適配器
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    
    # 創建提示模板
    template = """
    你是一位專業的財務分析師，專門研究企業破產預測。基於以下資訊，請回答用戶的問題。

    參考資料：
    {context}
    
    用戶問題：{question}
    
    請提供基於參考資料的詳細回答，引用相關的財務指標和公司治理指標。如果無法從資料中找到答案，請清楚說明。
    以繁體中文回答問題，若有專業術語需標注英文原文。
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # 創建 RetrievalQA 鏈
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # 使用 langchain 適配的 llm
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def query_rag_system(qa_chain, query):
    """
    使用 RAG 系統查詢問題
    
    Parameters:
    qa_chain (object): 配置好的 RetrievalQA 系統
    query (str): 用戶查詢問題
    
    Returns:
    str: 回答結果
    """
    try:
        result = qa_chain.run(query)
        return result
    except Exception as e:
        return f"查詢 RAG 系統時發生錯誤: {str(e)}"

# 使用範例
if __name__ == "__main__":
    # 創建 RAG 系統並保存向量資料庫
    persist_dir = "D:/Shawn/python_class/Tibame/Team-project/bankruptcy_prediction/src"
    rag_system = create_RAG_system(persist_directory=persist_dir)
    
    # # 範例查詢
    # query = "哪些財務比率對於破產預測最有效?"
    # result = query_rag_system(rag_system, query)
    # print(result)
    
    # query = "公司治理指標如何影響破產風險?"
    # result = query_rag_system(rag_system, query)
    # print(result)
