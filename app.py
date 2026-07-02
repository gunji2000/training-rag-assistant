from dotenv import load_dotenv
import os

load_dotenv()

# .envに記載したAPIキーを読み込む
api_key = os.getenv("GOOGLE_API_KEY")

# APIキーを表示
print(api_key)

# import chainlit as cl

# @cl.on_message
# async def main(message: cl.Message):
#     await cl.Message(
#         content=f"受け取ったよ: {message.content}"
#     ).send()