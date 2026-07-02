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
    google_api_key=api_key
)

# Geminiへ質問
response = llm.invoke("こんにちは")

# Geminiの回答を表示
print(response.content)

# import chainlit as cl

# @cl.on_message
# async def main(message: cl.Message):
#     await cl.Message(
#         content=f"受け取ったよ: {message.content}"
#     ).send()