import os
import chainlit as cl


from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

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

# 履歴の辞書
store = {}

# ユーザーIDがなければ枠を作る
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

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
    
    # ===== 確認用 =====
    print(f"チャンク数: {len(docs)}")

    for i, doc in enumerate(docs):
        print("-" * 50)
        print(f"Chunk {i}")
        print(doc.metadata)
        print(doc.page_content[:100])
    # ==================

    # ===== Embedding確認 =====
    text = "VPNとは何ですか？"

    vector = embeddings.embed_query(text)

    print("=" * 50)
    print("Embedding確認")
    print(f"型: {type(vector)}")
    print(f"ベクトルの次元数: {len(vector)}")
    print(f"先頭10個: {vector[:10]}")

    text1 = "VPNとは何ですか？"
    text2 = "VPNについて教えてください。"

    vector1 = embeddings.embed_query(text1)
    vector2 = embeddings.embed_query(text2)

    print(f"text1の先頭5個: {vector1[:5]}")
    print(f"text2の先頭5個: {vector2[:5]}")
    # =======================

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

    # ユーザーセッションの取得
    session_id = cl.context.session.id

    # 類似検索
    docs = vectorstore.similarity_search(query, k=3)
    print(f"検索結果数: {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"検索結果 {i}: {doc.page_content[:100]}...")

    # 検索結果を結合してコンテキストとして使用
    context = "\n\n".join([d.page_content for d in docs])

    # プロンプトテンプレートを使用してプロンプトを作成
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         以下の情報を参考にして回答してください。

         ### コンテキスト
         {context}
         """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_message_key="input",
        history_messages_key="history"
    )

    response = chain_with_history.invoke(
        {
            "context": context,
            "input": query,
        },
        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )

    # Geminiの回答を表示
    await cl.Message(content=response.content).send()

