from google import genai

client = genai.Client(api_key="YOUR_API_KEY")


def generate_thought(tasks, learned, unclear):
    prompt = f"""
次の内容をもとに、業務日報の「所感」を自然な文章で作ってください。

■本日やったこと
{tasks}

■理解できたこと
{learned}

■理解できなかったこと
{unclear}

・200〜300文字程度
・固すぎない
・エンジニアの日報っぽく
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    return response.text

def make_report(section1, section2):
    return f"""
==============================
📄 業務報告
==============================

(1)本日やったこと
{section1["tasks"]}

(2)理解できたこと
{section1["learned"]}

(3)理解できなかったこと
{section1["unclear"]}


==============================
🧠 所感
==============================

{section2["thought"]}
""".strip()


def main():
    print("==== 日報作成ツール ====")

    print("\n【業務報告を入力】")

    tasks = input("(1)本日やったこと（箇条書きOK）:\n")
    learned = input("\n(2)理解できたこと:\n")
    unclear = input("\n(3)理解できなかったこと:\n")

    print("\n【所感を入力】")
    thought = generate_thought(tasks, learned, unclear)

    section1 = {
        "tasks": tasks,
        "learned": learned,
        "unclear": unclear
    }

    section2 = {
        "thought": thought
    }

    report = make_report(section1, section2)

    print("\n\n===== 完成した日報 =====\n")
    print(report)


if __name__ == "__main__":
    main()

