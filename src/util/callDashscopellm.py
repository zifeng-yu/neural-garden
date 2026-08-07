import json
import logging
from typing import Any

import dashscope
from dashscope import MultiModalConversation

from src.config.config import API_KEY, LLM_MODEL
from src.util.llmException import LLMException

logger = logging.getLogger(__name__)


def generate(
    sys_prompt_content: str,
    user_prompt_content: str,
    temperature: float = 0.1,
    enable_thinking: bool = False,
    response_json: bool = False,
    response_class: type | None = None,
) -> Any:
    prompt = []
    system_prompt = {"role": "system", "content": f"{sys_prompt_content}"}
    user_prompt = {
        "role": "user",
        "content": f"{user_prompt_content}",
    }
    prompt.append(system_prompt)
    prompt.append(user_prompt)
    dashscope.api_key = API_KEY
    params = {
        "model": LLM_MODEL,
        "messages": prompt,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
    }
    if response_json:
        params["response_format"] = {"type": "json_object"}
    response = MultiModalConversation.call(**params)

    if response.status_code != 200:
        raise LLMException(f"{response.status_code}:{response.message}")
    text = response.output.choices[0].message.content[0]["text"]
    logger.info(text)
    if not response_json:
        return text
    data = json.loads(text)
    if response_class is not None:
        return response_class(**data)
    return data
