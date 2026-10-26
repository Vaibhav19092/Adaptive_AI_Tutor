"""The shared data that flows through the LangGraph workflow."""

import operator
from typing import Annotated, TypedDict


class TutorState(TypedDict, total=False):
    """One 'clipboard' passed from node to node.

    total=False means a key may be missing until some node fills it in.
    """

    question: str
    answer: str
    error: str
    # The reducer `operator.add` tells LangGraph to APPEND each node's
    # steps to the list instead of replacing it. This gives us a trace
    # proving which nodes actually ran.
    steps: Annotated[list[str], operator.add]