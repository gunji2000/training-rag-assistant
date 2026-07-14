from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    count: int


# カウント確認
def check_count(state: State):
    return state


# カウント増加
def increment(state: State):
    return {
        "count": state["count"] + 1
    }


# 条件分岐
def should_continue(state: State):
    if state["count"] < 3:
        return "increment"

    return "end"


graph_builder = StateGraph(State)


graph_builder.add_node("check_count", check_count)
graph_builder.add_node("increment", increment)


graph_builder.add_edge(
    START,
    "check_count"
)


graph_builder.add_conditional_edges(
    "check_count",
    should_continue,
    {
        "increment": "increment",
        "end": END
    }
)


graph_builder.add_edge(
    "increment",
    "check_count"
)


graph = graph_builder.compile()


result = graph.invoke(
    {
        "count": 0
    }
)


print(result)