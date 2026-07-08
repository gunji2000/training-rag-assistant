# 研修用AI（RAGチャットボット）

## 📌 概要
本プロジェクトは、社内研修資料を対象としたRAG（Retrieval-Augmented Generation）チャットボットの構築を目的とする。
ユーザーの質問に対して、PDFなどのドキュメントから関連情報を検索し、LLMを用いて要約・回答を生成する。

PDFドキュメントをアップロードすることで、資料内容を検索対象とし、自然言語による質問応答を可能とするシステムを構築した。

また、会話履歴保持や回答元ドキュメントの出典表示など、実用性を考慮した機能を追加した。

---

## 🎯 目的
- 社内研修資料の検索性向上
- 自然言語でのドキュメント検索の実現
- RAG構成（Embedding + Vector Search + LLM）の理解と実装
- 実務レベルのAIアプリケーション構築スキル習得

---

## 🧱 使用技術
- Python
- LangChain
- Gemini API（Google Generative AI）
- Chroma（ベクトルデータベース）
- Chainlit（チャットUI）
- PyPDFLoader（PDF解析）
- Git / GitHub（バージョン管理）

---

## 🏗️ アーキテクチャ（概要）

ユーザー入力  
→ Chainlit（UI）  
→ PDFアップロード  
→ PyPDFLoader（ドキュメント解析）  
→ Chunk分割  
→ Embedding（ベクトル化）  
→ Chroma（ベクトル検索）  
→ Retriever（関連情報取得）  
→ Gemini API（回答生成）  
→ 応答表示  

---

## 📁 ディレクトリ構成（予定）

training-rag-assistant/
    app.py
    config.py
    data/
        pdf/
    .venv/
    .env
    .gitignore
    README.md
    chainlit.md

---

## ✨ 実装機能

### RAG機能
- PDFドキュメント読み込み
- Document Loaderによるデータ取得
- RecursiveCharacterTextSplitterによるChunk分割
- Embeddingによるベクトル化
- Chromaへのベクトル登録
- Retrieverによる類似検索
- 検索結果を利用した回答生成

### チャット機能
- Gemini API連携
- 会話履歴保持
- 質問内容に応じた回答生成

### UI・管理機能
- ChainlitによるチャットUI
- PDFアップロード対応
- 回答元ドキュメントの出典表示
- config.pyによる設定値管理

### 画像対応（開発中）
- Chainlitによる画像入力取得
- 画像ファイルパス取得
- Base64形式への変換処理

---

## 🚧 現在の進捗
- 開発環境構築完了
- GitHub連携完了
- Chainlit動作確認（Echo Bot）
- RAG設計フェーズ完了
- Gemini API連携
- PDFドキュメント対応
- Retriever実装
- Vector DB（Chroma）連携
- RAGパイプライン構築完了
- 会話履歴実装
- 出典表示実装
- 画像入力対応（Base64変換まで）

---

## 🔜 今後の開発予定
- Gemini Vision API連携
- 画像を用いた質問応答
- 精度改善・評価
- Retriever検索精度改善
- RAG回答精度の評価

---

## 📝 補足
本プロジェクトは学習目的を含むRAG実装であり、段階的に機能を拡張している。

また、過去に経験したC++によるベクトル検索基盤の高速化検証と、PythonによるRAGアプリケーション構築の知識をつなげることを意識して開発を行っている。
