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

- LangGraphによるRAG Workflow構築
- StateによるNode間データ共有
- Retrieve Node
- Generation Node
- Source Node
- Conditional Edgeによる条件分岐
- NoAnswer Nodeによる検索失敗時の処理

### チャット機能

- Gemini API連携
- 会話履歴保持
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

- LangGraph Memoryへの移行
- RunnableWithMessageHistoryの置き換え
- Retriever検索精度改善
- Query Rewrite
- Reranker導入
- FastAPIによるAPI化
- SQLiteによる会話履歴永続化
- Docker環境構築
- Gemini Vision対応
- マルチモーダルRAG
- Agentic RAGへの発展

---

## 📝 補足

本プロジェクトは、実務で利用されるAIアプリケーションを想定し、段階的に機能を拡張している。

現在はLangChainによる基本的なRAG構成に加え、LangGraphを用いたワークフロー管理を導入し、State・Node・Edgeによる処理の分離や条件分岐を実装している。

今後はFastAPIによるAPI化、LangGraph Memoryによる会話履歴管理、Docker環境構築などを追加し、より実務に近いAIアプリケーションへ発展させる予定である。