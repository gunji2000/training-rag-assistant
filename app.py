import os
import chainlit as cl


from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

# .envに記載したAPIキーを読み込む
api_key = os.getenv("GOOGLE_API_KEY")


# Geminiとの接続準備
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

# グローバル保存(簡易RAG)
vectorstore = None

# 初期化
@cl.on_chat_start
async def on_chat_start():

    global vectorstore

    # PDFロード
    loader = PyPDFLoader("data/test.pdf")
    documents = loader.load()

    # chunk分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)
    
    # ベクトル化
    vectorstore = Chroma.from_documents(
        docs,
        embeddings
    )

    await cl.Message(content="RAG初期化完了 (PDF読み込み済み)").send()

# メイン処理
@cl.on_message
async def main(message: cl.Message):
    global vectorstore
    query = message.content

    # 類似検索
    docs = vectorstore.similarity_search(query, k=3)

    # 検索結果を結合してコンテキストとして使用
    context = "\n\n".join([d.page_content for d in docs])

    # プロンプト
    prompt = f"""
以下の情報を参考にして回答してください。
### コンテキスト
{context}
### 質問
{query}
    """
    response = llm.invoke(prompt)

    # Geminiの回答を表示
    await cl.Message(content=response.content).send()

# # Geminiへ質問
# response = llm.invoke("こんにちは")

# # Geminiの回答を表示
# print(response.content)

# # RAGの例
# @cl.on_message
# async def main(message: cl.Message):
#     response = llm.invoke(message.content)
#     await cl.Message(content=response.content).send()
