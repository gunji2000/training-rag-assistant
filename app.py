import base64
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
from google import genai
from google.genai import types

from config import (
    MODEL_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEARCH_K,
    SYSTEM_PROMPT,
)

# ==============================
# 環境設定・API接続設定
# ==============================
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")       # .envに記載したAPIキーを読み込む
client = genai.Client(api_key=api_key)

# Geminiとの接続準備
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=api_key
)
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=api_key
)

# ==============================
# LLM Chain・会話履歴管理設定
# ==============================
prompt = ChatPromptTemplate.from_messages([     # プロンプトテンプレートを使用してプロンプトを作成
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
chain = prompt | llm        # プロンプトとLLMまでのパイプラインを作成
retriever = None            # グローバル保存(簡易RAG)

store = {}                  # 履歴の辞書
def get_session_history(session_id: str) -> BaseChatMessageHistory:     # ユーザーIDがなければ枠を作る
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(        # 履歴付きのRunnableを作成
    chain,
    get_session_history,
    input_message_key="input",
    history_messages_key="history"
)

# ==============================
# RAG初期化処理（Document → Retriever作成）
# ==============================
def load_documents(file_path):      # PDFファイルを読み込みDocumentを取得する関数
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def load_all_documents(file_dir):
    all_documents = []
    file_list = os.listdir(file_dir)
    for file_name in file_list:
        file_path = os.path.join(file_dir, file_name)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_documents += documents
    return all_documents

def split_documents(documents):     # Documentを分割する関数
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )
    docs = splitter.split_documents(documents)
    return docs

def create_vectorstore(docs, embeddings):   # 分割したDocumentをEmbeddingしてVectorStoreを作成する関数
    vectorstore = Chroma.from_documents(
        docs,
        embeddings
    )
    return vectorstore

def create_retriever(vectorstore):          # VectorStoreから検索用のRetrieverを作成する関数
    retriever = vectorstore.as_retriever(
        search_kwargs = {"k": SEARCH_K}
    )
    return retriever

# ==============================
# RAG検索・回答生成処理
# ==============================
def search_documents(retriever, query):     # Retrieverを使用して類似検索を実行する関数
    docs = retriever.invoke(query)
    return docs

def create_context(docs):       # 検索結果を結合してコンテキストとして使用
    context = "\n\n".join([d.page_content for d in docs])
    return context

def generate_answer(chain_with_history, docs, query, session_id):   # 検索結果をコンテキスト形式に変換し、LLMへ問い合わせる関数
    response = chain_with_history.invoke(
            {
                "context": create_context(docs),
                "input": query,
            },
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )
    return response

# ==============================
# 画像処理
# ==============================
def encode_image_to_base64(image_path):
    """
    画像ファイルを読み込みBase64形式へ変換する
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_base64 = base64.b64encode(image_bytes)

    return image_base64.decode("utf-8")

def analyze_image(image_path, mime_type, query):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            ),
            query
        ]
    )

    return response.text

# ==============================
# 回答整形・出典情報作成処理
# ==============================
def create_sources(docs):       # 検索結果の出典を作成
    sources = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page_label", "?")
        sources.add((source, page))
    return sources

def create_answer_with_sources(response, docs):     # LLMの回答に検索元情報を追加する関数
    answer = response.content
    answer += "\n\n---\n参考文献:\n"

    for source, page in sorted(create_sources(docs)):
        answer += f"- {source} ({page}ページ)\n"
    return answer

# ==============================
# Chainlitイベント処理
# ==============================
@cl.on_chat_start       # 初期化
async def on_chat_start():
    global retriever

    documents = load_all_documents("data")
    # chunk分割
    docs = split_documents(documents)
    # ベクトル化
    vectorstore = create_vectorstore(docs, embeddings)
    # Retriever作成
    retriever = create_retriever(vectorstore)

    await cl.Message(content="RAG初期化完了 (PDF読み込み済み)").send()

@cl.on_message          # メイン処理
async def main(message: cl.Message):
    query = message.content

    if message.elements:
        image = message.elements[0]

        response = analyze_image(
            image.path,
            image.mime,
            query
        )

        await cl.Message(content=response).send()
        return
    
    # ユーザーセッションの取得
    session_id = cl.context.session.id
    # 類似検索
    docs = search_documents(retriever, query)
    # 回答を取得
    try:
        response = generate_answer(chain_with_history, docs, query, session_id)
    except Exception as e:
        print(f"エラー: {e}")
        await cl.Message(content="エラーが発生しました").send()
        return

    # ==Retriever検索結果について出力する==
    debug_text = "\n\n---\n Retriever検索結果:\n"
    for i, doc in enumerate(docs):
        debug_text += f"\nChunk {i}: {doc.page_content[:100]}\n\n"
    # =================================

    # 回答に対する出典を作成
    answer = create_answer_with_sources(response, docs)

    # Geminiの回答を表示
    await cl.Message(content=debug_text).send() # debug用
    await cl.Message(content=answer).send()

