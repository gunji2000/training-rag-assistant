# 研修用AI（RAGチャットボット）

## 📌 概要
本プロジェクトは、社内研修資料を対象としたRAG（Retrieval-Augmented Generation）チャットボットの構築を目的とする。
ユーザーの質問に対して、PDFなどのドキュメントから関連情報を検索し、LLMを用いて要約・回答を生成する。

最終的には、社内ドキュメント検索AIとして、用語検索や資料参照を自然言語で行えるシステムを目指す。

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
→ LangChain（処理フロー）  
→ Embedding（ベクトル化）  
→ Chroma（検索）  
→ Gemini API（回答生成）  
→ 応答表示  

---

## 📁 ディレクトリ構成（予定）

training-rag-assistant/
    app.py
    rag/
        loader.py
        retriever.py
        vectorstore.py
    data/
        pdf/
    .venv/
    .env
    .gitignore
    README.md
    chainlit.md

---

## 🚧 現在の進捗
- 開発環境構築完了
- GitHub連携完了
- Chainlit動作確認（Echo Bot）
- RAG設計フェーズ完了
- Gemini API連携

---

## 🔜 今後の開発予定
- RAGパイプライン構築
- PDFドキュメント対応
- Retriever実装
- 精度改善・評価

---

## 📝 補足
本プロジェクトは学習目的を含むRAG実装であり、段階的に機能を拡張していく。