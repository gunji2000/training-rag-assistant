import os
from dotenv import load_dotenv
from typing import Any
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from config import (
    MODEL_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEARCH_K,
    SYSTEM_PROMPT,
)

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=api_key
)
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=api_key
)
prompt = ChatPromptTemplate.from_messages([     # プロンプトテンプレートを使用してプロンプトを作成
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
chain = prompt | llm 

class RAGState(TypedDict):
    query: str
    session_id: str
    docs: list[Document]
    # context: str
    answer: Any

def retrieve_node(state: RAGState):
    print("=== Retrieve Node ===")
    print(f"query: {state['query']}")
    print(f"docs(before): {len(state['docs'])}")

    docs = search_documents(
        retriever,
        state["query"]
    )
    print(f"docs(after): {len(docs)}")

    return {
        "docs": docs
    }

def generation_node(state: RAGState):
    print("=== Generation Node ===")
    print(f"query: {state['query']}")
    print(f"docs: {len(state['docs'])}")

    response = generate_answer(
        chain_with_history,
        state["docs"],
        state["query"],
        state["session_id"]
    )
    print(f"answer type: {type(response).__name__}")
    print(f"answer: {response.content}")

    return {
        "answer": response
    }

def source_node(state: RAGState):
    print("=== Source Node ===")
    print(f"docs: {len(state['docs'])}")
    print(f"answer(before): {type(state['answer']).__name__}")

    answer = create_answer_with_sources(
        state["answer"],
        state["docs"]
        )
    print(f"answer(after): {type(answer).__name__}")
    print(f"answer(after): {answer}")
    
    return {
        "answer": answer
    }


def load_all_documents(file_dir: str) -> list[Document]:
    all_documents = []
    file_list = os.listdir(file_dir)
    for file_name in file_list:
        file_path = os.path.join(file_dir, file_name)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_documents += documents
    return all_documents

def split_documents(documents: list[Document]) -> list[Document]:     # Documentを分割する関数
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )
    docs = splitter.split_documents(documents)
    return docs

def create_vectorstore(docs: list[Document], embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:   # 分割したDocumentをEmbeddingしてVectorStoreを作成する関数
    """
    分割済みDocumentをEmbeddingし、
    ChromaのVectorStoreを作成する。
    """
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

def search_documents(retriever: VectorStoreRetriever, query: str) -> list[Document]:     # Retrieverを使用して類似検索を実行する関数
    docs = retriever.invoke(query)
    return docs

def create_context(docs: list[Document]) -> str:       # 検索結果を結合してコンテキストとして使用
    context = "\n\n".join([d.page_content for d in docs])
    return context

def generate_answer(chain_with_history, docs: list[Document], query: str, session_id: str) -> Any:   # 検索結果をコンテキスト形式に変換し、LLMへ問い合わせる関数
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

store = {}                  # 履歴の辞書
def get_session_history(session_id: str) :     # ユーザーIDがなければ枠を作る
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def create_sources(docs: list[Document]):       # 検索結果の出典を作成
    sources = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page_label", "?")
        sources.add((source, page))
    return sources
def create_answer_with_sources(response: Any, docs: list[Document]) -> str:     # LLMの回答に検索元情報を追加する関数
    answer = response.content
    answer += "\n\n---\n参考文献:\n"

    for source, page in sorted(create_sources(docs)):
        answer += f"- {source} ({page}ページ)\n"
    return answer

# 初期化
documents = load_all_documents("data")
docs = split_documents(documents)
vectorstore = create_vectorstore(docs, embeddings)
retriever = create_retriever(vectorstore)
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_message_key="input",
    history_messages_key="history"
)

graph_builder = StateGraph(RAGState)

graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generation_node)
graph_builder.add_node("source", source_node)

graph_builder.add_edge(
    START,
    "retrieve"
)

graph_builder.add_edge(
    "retrieve",
    "generate"
)

graph_builder.add_edge(
    "generate",
    "source"
)

graph_builder.add_edge(
    "source",
    END
)

graph = graph_builder.compile()

result = graph.invoke(
    {
        "query": "RAGとは？",
        "session_id": "test",
        "docs": [],
        # "context": "",
        "answer": None
    }
)

print(result)