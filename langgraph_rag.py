import os
import chainlit as cl

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
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage

from typing import TypedDict
from langgraph.graph import MessagesState
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
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])
chain = prompt | llm 

class RAGState(MessagesState):
    query: str
    docs: list[Document]
    answer: Any

def retrieve_node(state: RAGState):
    print("===== Retrieve =====")
    print(state["messages"])
    print(state["query"])
    print(state["docs"])

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
    print("messages:")
    for message in state["messages"]:
        print(type(message).__name__, message.content)

    response = generate_answer(
        chain,
        state["docs"],
        state["query"],
        state["messages"]
    )
    print(f"answer type: {type(response).__name__}")
    print(f"answer: {response.content}")

    return {
        "answer": response,
        "messages": [response]
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

def no_answer_node(state):
    return {
        "answer": "検索結果がありませんでした。"
    }


def should_generate(state: RAGState):
    if state["query"] == "終了":
        return "end"
    return "generate"


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

def generate_answer(chain, docs: list[Document], query: str, messages) -> Any:   # 検索結果をコンテキスト形式に変換し、LLMへ問い合わせる関数
    response = chain.invoke(
        {
            "context": create_context(docs),
            "input": query,
            "messages": messages
        }
    )
    return response

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

graph_builder = StateGraph(RAGState)

graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generation_node)
graph_builder.add_node("source", source_node)
graph_builder.add_node("no_answer", no_answer_node)

graph_builder.add_edge(
    START,
    "retrieve"
)

graph_builder.add_conditional_edges(
    "retrieve",
    should_generate,
    {
        "end": "no_answer",
        "generate": "generate"
    }
)

graph_builder.add_edge(
    "no_answer",
    END
)

graph_builder.add_edge(
    "generate",
    "source"
)

graph_builder.add_edge(
    "source",
    END
)

memory = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=memory
)


@cl.on_message
async def main(message):
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=message.content)
            ],
            "query": message.content,
            "docs": [],
            "answer": None
        },
        config={
            "configurable": {
                "thread_id": cl.context.session.id
            }
        }
    )

    await cl.Message(
        content=result["answer"]
    ).send()