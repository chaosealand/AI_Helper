# 考前小救星 - RAG 複習助教

## 簡介
此專案使用 RAG + 向量資料庫與 Gradio 提供互動式問答與自動出題功能。適合在 Windows 環境下執行並上傳教材檔案（`.txt`、`.pdf`、`.docx`），支援上傳多個檔案。

## 目錄結構（重要檔案）
- `main.py`：主介面以及程式入口（Gradio）
- `RAG_Core.py`、`Vector_DataBase.py`：核心邏輯
- `requirements.txt`：相依套件
- `example_data/`：範例上傳檔案
- `README.md`：使用說明
- `TOKEN.py` : 存放 API 

## 使用方法
1. 依照 `main.py` 裡的說明，安裝好套件後，執行 `main.py`。
2. 在 Gradio 介面上傳教材檔案。
3. 選擇功能（問答或出題），並輸入相關問題或參數。
4. 點擊執行，等待結果顯示。

## 注意事項
- `TOKEN.py`內的API TOKEN已被移除，需要重新填入Grok和HuggingFace TOKEN才可以運行
- 優先依照 `main.py` 裡的說明安裝相依套件，若有問題可參考 `requirements.txt`。
- 模型需要比較多時間跑，錄影時會跳過等待時間
## 介紹Youtube連結
https://youtu.be/MnoIJO0eaG4
