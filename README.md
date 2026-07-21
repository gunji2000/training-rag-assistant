# 研修用AI（RAGチャットボット）

## 📌 概要

本プロジェクトは、社内研修資料を対象としたRAG（Retrieval-Augmented Generation）チャットボットの構築を目的とする。

ユーザーの質問に対して、PDFなどのドキュメントから関連情報を検索し、LLMを用いて回答を生成する。

PDFドキュメントを検索対象とし、自然言語による質問応答を実現するとともに、会話履歴保持や回答元ドキュメントの出典表示など、実用性を考慮した機能を実装している。

また、LangChainによる基本的なRAGを実装した後、同等の処理をLangGraphへ移行し、State・Node・Edgeによるワークフロー管理や条件分岐の実装を進めている。

---

## 🎯 目的

- 社内研修資料の検索性向上
- 自然言語によるドキュメント検索の実現
- RAG（Embedding + Vector Search + LLM）の理解と実装
- LangGraphを用いたAIワークフロー構築の理解
- 実務レベルのAIアプリケーション開発スキルの習得

---

## 🧱 使用技術

- Python
- LangChain
- LangGraph
- Gemini API（Google Generative AI）
- Chroma（Vector Database）
- Chainlit
- FastAPI（学習・実装中）
- SQLite（学習・実装中）
- PyPDFLoader
- Git / GitHub

---

## 🏗️ アーキテクチャ（概要）

```
ユーザー入力
        │
        ▼
   Chainlit
        │
        ▼
LangGraph Workflow
        │
        ▼
 Retrieve Node
        │
        ▼
Conditional Edge
   ├─────────────┐
   ▼             ▼
Generation    NoAnswer
   │
   ▼
Source
   │
   ▼
Gemini回答 + 出典表示
```

---

## 📁 ディレクトリ構成

```
training-rag-assistant/

├── app.py              # LangChain版RAG
├── langgraph_rag.py    # LangGraph版RAG
├── config.py
├── data/
├── chainlit.md
├── README.md
├── .env
└── .gitignore
```

---

## ✨ 実装機能

### RAG機能

- PDFドキュメント読み込み
- Document Loaderによるデータ取得
- RecursiveCharacterTextSplitterによるChunk分割
- Gemini Embeddingによるベクトル化
- Chromaへのベクトル登録
- Retrieverによる類似検索
- 検索結果を利用した回答生成

### LangGraph

- StateGraphによるワークフロー構築
- MessagesStateによる会話履歴管理
- StateによるNode間データ共有
- Node単位での責務分離
- Edgeによる処理フロー管理
- Conditional Edgeによる条件分岐
- InMemorySaver(Checkpointer)によるState保存
- thread_idによる会話管理

### チャット機能

- Gemini API連携
- LangGraph Memoryによる会話履歴管理
- MessagesStateを利用したメッセージ保持
- InMemorySaver(Checkpointer)によるState管理
- thread_id単位での会話履歴復元
- 質問内容に応じた回答生成

### UI・管理機能

- ChainlitによるチャットUI
- LangGraph WorkflowのChainlit連携
- PDFアップロード対応
- 回答元ドキュメントの出典表示
- config.pyによる設定値管理

### 画像対応（開発中）

- Chainlitによる画像入力取得
- 画像ファイル読み込み
- Base64形式への変換処理
- Gemini Vision対応準備


## 💾 LangGraph Memory

本プロジェクトではLangGraph Memoryを利用して会話履歴を管理している。

- MessagesStateを継承したStateを利用
- InMemorySaverをCheckpointerとして利用
- thread_id単位でStateを保存・復元
- Reducerによりmessagesを上書きではなく追加して保持

これにより、RunnableWithMessageHistoryを利用せずにLangGraph標準のMemory管理へ移行している。

---

## 🚧 現在の進捗

- 開発環境構築
- GitHub連携
- Chainlit動作確認
- Gemini API連携
- PDFドキュメント対応
- Retriever実装
- Chroma連携
- RAGパイプライン構築
- 会話履歴実装
- 回答元ドキュメントの出典表示
- 画像入力対応（Base64変換）
- LangGraphによるWorkflow化
- State実装
- Node実装
- Edge実装
- Conditional Edge実装
- NoAnswer Node実装
- ChainlitからLangGraph Workflow実行
- FastAPI・SQLite学習開始

---

## 🔜 今後の開発予定

- Retriever検索精度改善
- Similarity Threshold導入
- Query Rewrite
- Reranker導入
- LLM評価
- FastAPIによるAPI化
- SQLiteによる履歴永続化
- Docker環境構築
- Gemini Vision対応
- Agentic RAGへの発展

---

## 📝 補足

本プロジェクトは、実務で利用されるAIアプリケーションを想定し、段階的に機能を拡張している。

現在はLangGraphを中心とした構成へ移行し、

- State
- Node
- Edge
- Conditional Edge
- LangGraph Memory

を利用したワークフロー管理を実装している。

また、MessagesState・Reducer・Checkpointerを利用した会話履歴管理へ移行し、LangGraph標準のMemory管理を採用している。