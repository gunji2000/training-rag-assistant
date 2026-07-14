from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# State定義
class State(TypedDict):
    message: str
    count: int


# Node①
def add_text(state: State) -> State:
    return {
        "message": state["message"] + " LangGraph",
        "count": state["count"] + 1
    }


# Node②
def to_upper(state: State) -> State:
    return {
        "message": state["message"].upper()
    }


# Graph作成
graph_builder = StateGraph(State)


# Node登録
graph_builder.add_node("add_text", add_text)
graph_builder.add_node("to_upper", to_upper)


# Edge設定
graph_builder.add_edge(START, "add_text")
graph_builder.add_edge("add_text", "to_upper")
graph_builder.add_edge("to_upper", END)


# Compile
graph = graph_builder.compile()


# 実行
result = graph.invoke(
    {
        "message": "hello",
        "count": 0
    }
)

print(result)