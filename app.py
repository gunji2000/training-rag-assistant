import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# .envに記載したAPIキーを読み込む
api_key = os.getenv("GOOGLE_API_KEY")

# APIキーを表示
# print(api_key)

# Geminiとの接続準備
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    system_instruction="あなたは丁寧で簡潔に答えるアシスタントです"
)

# # Geminiへ質問
# response = llm.invoke("こんにちは")

# # Geminiの回答を表示
# print(response.content)


@cl.on_message
async def main(message: cl.Message):
    response = llm.invoke(message.content)
    await cl.Message(content=response.content).send()
