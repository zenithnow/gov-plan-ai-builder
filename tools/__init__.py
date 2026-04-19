from .evaluation_score_checker import TOOL_DEFINITION as EVAL_TOOL, handle_tool_call as eval_handler
from .market_size_calculator import TOOL_DEFINITION as MARKET_TOOL, handle_tool_call as market_handler
from .style_refiner import TOOL_DEFINITION as STYLE_TOOL, handle_tool_call as style_handler

ALL_TOOLS = [EVAL_TOOL, MARKET_TOOL, STYLE_TOOL]

TOOL_HANDLERS = {
    "evaluation_score_checker": eval_handler,
    "market_size_calculator": market_handler,
    "style_refiner": style_handler,
}
