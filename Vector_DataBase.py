import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import login
from langchain_community.embeddings import HuggingFaceEmbeddings
import TOKEN

class EmbeddingGemmaEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="google/embeddinggemma-300m",   # HF 上的官方模型
            encode_kwargs={"normalize_embeddings": True},  # 一般檢索慣例
            **kwargs
        )

    def embed_documents(self, texts):
        # 文件向量：title 可用 "none"，或自行帶入檔名/章節標題以微幅加分
        texts = [f'title: none | text: {t}' for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        # 查詢向量：官方建議的 Retrieval-Query 前綴
        return super().embed_query(f'task: search result | query: {text}')


def CreateDataBase (file_names) :

    if not file_names:
        raise ValueError("❌ file_names 不能為空")
    folder_path = "uploaded_docs"
    documents = []
    for file_name in file_names:
        path = os.path.join(folder_path, file_name)
        if not os.path.exists(path):
            print(f"⚠️ 檔案不存在，跳過：{file_name}")
            continue

        if file_name.endswith(".txt"):
            loader = TextLoader(path)
        elif file_name.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file_name.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
        else:
            print(f"⚠️ 不支援的檔案格式，跳過：{file_name}")
            continue

        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"❌ 載入失敗：{file_name}, 錯誤：{e}")
            continue

        if not documents:
            raise ValueError("❌ 沒有成功載入任何文件")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    # %%

    hf_token = TOKEN.HuggingFaceToken
    # %%

    login(token=hf_token)

    embedding_model = EmbeddingGemmaEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embedding_model)

    vectorstore.save_local("faiss_db")

