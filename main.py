import gradio as gr
from pathlib import Path
import RAG_Core
import Vector_DataBase
import shutil



'''
需安裝以下套件
!pip install gradio
!pip install -U langchain langchain-community pypdf python-docx faiss-cpu
!pip install -U sentence-transformers transformers
!pip install -U langchain langchain-community faiss-cpu transformers sentence-transformers huggingface_hub
!pip -q install "aisuite[all]"
'''


def normalize_latex(text: str) -> str:
    return (
        text.replace("\\[", "$$")
            .replace("\\]", "$$")
            .replace("\\(", "$")
            .replace("\\)", "$")
            .replace("$$$", "$$")
            .replace(" ,", " \\cdot ")
    )


def upload_and_create_db(files):
    if not files:
        return "請先上傳檔案"

    upload_dir = Path("uploaded_docs")
    upload_dir.mkdir(exist_ok=True)

    saved_files = []
    for file in files:
        # Gradio 傳遞的是暫存檔案路徑（字串）
        src_path = Path(file) if isinstance(file, str) else Path(file.name)
        dest_path = upload_dir / src_path.name

        # 直接複製檔案
        shutil.copy(src_path, dest_path)
        saved_files.append(dest_path.name)

    # 呼叫向量資料庫建立函式
    try:
        Vector_DataBase.CreateDataBase(saved_files)
        status = f"已上傳 {len(saved_files)} 個檔案並建立資料庫\n"
        RAG_Core.initialize_vectorstore()
    except Exception as e:
        return f"建立資料庫失敗：{e}"

    return status + "\n".join(saved_files)


def respond(message, chat_history):
    """RAG 問答回應"""
    if not message.strip():
        return chat_history

    response = RAG_Core.chat_with_rag(message)
    response = normalize_latex(response)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})
    return chat_history


def generate_quiz_ui(topic, num_q):
    """生成測驗題目"""
    quiz = RAG_Core.generate_quiz_from_rag(topic, num_q)
    quiz = normalize_latex(quiz)
    return quiz

with gr.Blocks() as demo:
    gr.HTML("""
        <style>
        #rag-chatbot .message-content h1,
        #rag-chatbot .message-content h2,
        #rag-chatbot .message-content h3 {
            font-size: 1rem;
            margin: 0.4rem 0;
        }
        #rag-chatbot .message-content p {
            font-size: 0.95rem;
            line-height: 1.5;
            margin: 0.25rem 0;
        }
        </style>
        """)
    gr.Markdown("# 考前小救星 - RAG 複習助教")

    # 檔案上傳區
    with gr.Accordion("步驟一：上傳教材檔案", open=True):
        file_upload = gr.File(
            label="支援 .txt / .pdf / .docx",
            file_count="multiple",
            file_types=[".txt", ".pdf", ".docx"]
        )
        upload_btn = gr.Button("建立知識庫", variant="primary")
        upload_status = gr.Textbox(label="上傳狀態", interactive=False)

        upload_btn.click(
            upload_and_create_db,
            inputs=[file_upload],
            outputs=[upload_status]
        )

    # 主要功能區
    with gr.Tabs():
        # Tab 1: RAG 問答
        with gr.Tab("RAG 問答"):
            chatbot = gr.Chatbot(height=400, label="考前問答區",  latex_delimiters=[  # 在 Chatbot 元件啟用 LaTeX
            {"left": "$$", "right": "$$", "display": True},
            {"left": "$", "right": "$", "display": False}
        ])
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="例如：什麼是等價關係？",
                    label="輸入你的問題",
                    scale=4,

                )
                submit_btn = gr.Button("送出", scale=1)

            clear_btn = gr.Button("清除對話記錄")

            submit_btn.click(respond, [msg, chatbot], [chatbot])
            msg.submit(respond, [msg, chatbot], [chatbot])
            clear_btn.click(lambda: [], None, chatbot)

        # Tab 2: 測驗生成
        with gr.Tab("自動出小測驗"):
            gr.Markdown("### 根據教材內容自動出題（附參考解答）")

            with gr.Row():
                topic_box = gr.Textbox(
                    label="想複習的主題（可留空）",
                    placeholder="例如：離散中的等價關係",
                    scale=3
                )
                num_q_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="題目數量",
                    scale=1
                )

            generate_btn = gr.Button("產生小測驗", variant="primary")
            quiz_output = gr.Markdown(
                label="小測驗題目與參考解答",
                value="",  # 初始值為空字串
                latex_delimiters=[  # 在 Chatbot 元件啟用 LaTeX
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False}
                ]
            )

            generate_btn.click(
                generate_quiz_ui,
                inputs=[topic_box, num_q_slider],
                outputs=[quiz_output]
            )

if __name__ == "__main__":

    demo.launch()
