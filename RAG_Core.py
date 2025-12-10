from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from huggingface_hub import login
import TOKEN
import aisuite as ai

hf_token = TOKEN.HuggingFaceToken
login(token=hf_token)

api_key = TOKEN.GrokToken
os.environ["GROQ_API_KEY"] = api_key

model = "groq:openai/gpt-oss-120b"
client = ai.Client()

retriever = None

class EmbeddingGemmaEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="google/embeddinggemma-300m",
            encode_kwargs={"normalize_embeddings": True},
            **kwargs
        )

    def embed_documents(self, texts):
        texts = [f"title: none | text: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"task: search result | query: {text}")


embedding_model = EmbeddingGemmaEmbeddings()


def initialize_vectorstore():
    global retriever
    vectorstore = FAISS.load_local(
        "faiss_db",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})




# （1）一般 RAG 問答用的 system prompt
system_prompt_chat = """
你是一位大學「考前小救星」助教，專門幫學生複習某一門課（例如離散數學、線性代數、資料結構）。

請根據提供的教材內容回答問題：
- 儘量用台灣大學生習慣的中文說明
- 先給結論，再用步驟或條列式解釋
- 必要時舉 1 個簡單小例子幫助理解
- 如果資料不夠，請老實說「從目前資料看不出來」，不要亂掰。
"""

prompt_template_chat = """
以下是這門課（例如：離散、線代、資料結構）的部分教材內容：

{retrieved_chunks}

---

學生的問題是：

{question}

請根據上述「教材內容」來回答，不要使用外部沒出現過的定理。
回答格式建議：
1. 先一句話總結重點
2. 再用條列式說明步驟 / 定義 / 性質
3. 若適用，可以補一個小例子（簡單就好）
"""

# （2）自動出「小測驗題」的 system prompt
system_prompt_quiz = """
你是一位大學課程的助教，正在幫學生準備考前小測驗。

請根據提供的「課程教材內容」來命題：
- 題目難度：期中 / 期末考中等難度
- 題型：以簡答題、計算題、證明 / 說明題為主，可以少量選擇題
- 每一題一定要有「題目 (Q)」和「參考解答 (A)」
- 題目一定要可以從提供的內容推得出來，不要胡亂新增沒出現過的理論。
"""

prompt_template_quiz = """
以下是這門課的教材內容節錄：

{retrieved_chunks}

---

請根據這些內容，設計 {num_questions} 題考前小測驗。
請遵守：
- 題目盡量對應這些內容的重要觀念、定義、定理或典型例題
- 可以混合簡答 / 計算 / 證明 / 選擇題，但要適合考前複習
- 每題都要有參考解答，寫在 A 後面

輸出格式請嚴格使用：

Q1: （題目）
A1: （參考解答）

Q2: （題目）
A2: （參考解答）

...
"""


chat_history = []  # 若之後要做「有記憶的對話」可以用，目前先保留

def chat_with_rag(user_input: str) -> str:
    """
    一般 RAG 問答：根據使用者問題，從向量庫取出相關內容，讓 LLM 回答。
    """
    docs = retriever.invoke(user_input)  # 新版寫法

    retrieved_chunks = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt_template_chat.format(
        retrieved_chunks=retrieved_chunks,
        question=user_input,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt_chat},
            {"role": "user", "content": final_prompt},
        ],
    )
    answer = response.choices[0].message.content
    chat_history.append((user_input, answer))
    return answer


def generate_quiz_from_rag(topic: str, num_questions: int = 5) -> str:
    """
    自動出小測驗題目＋參考解答。
    topic 可以是：
      - 空字串：代表「整門課的小複習」
      - 關鍵字：例如 "equivalence relation", "DFS", "Eigenvalue"
    """
    if not topic.strip():
        query = "這門課的核心重點與常考觀念"
    else:
        query = topic

    docs = retriever.invoke(query)
    if not docs:
        return "資料不存在"

    retrieved_chunks = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt_template_quiz.format(
        retrieved_chunks=retrieved_chunks,
        num_questions=num_questions,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt_quiz},
            {"role": "user", "content": final_prompt},
        ],
    )
    quiz_text = response.choices[0].message.content
    return quiz_text
