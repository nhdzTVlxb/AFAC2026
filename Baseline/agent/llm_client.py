"""OpenAI兼容LLM客户端 - 支持自动记录合规日志"""
import os
import time
import json
import logging
import requests
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class LLMClient:
    """
    大模型API客户端

    支持OpenAI格式API调用。当API不可用时返回空字符串/空字典，不抛出异常。
    自动记录所有调用到call_history，可导出合规日志。
    """

    def __init__(self, model: str = "qwen-turbo", api_key: str = "",
                 base_url: str = "", temperature: float = 0.7,
                 max_tokens: int = 4096, timeout: int = 120):
        self.model = model
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.call_history: List[Dict[str, Any]] = []

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        """
        调用LLM进行对话

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认最大token数

        Returns:
            LLM生成的文本，失败时返回空字符串
        """
        if not self.api_key or not self.base_url:
            return ""

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        start = time.time()
        try:
            result = self._call_api(messages, temp, max_tok)
            latency = time.time() - start
            self.call_history.append({
                "timestamp": time.time(),
                "model": self.model,
                "latency": latency,
                "success": result is not None and len(result) > 0,
                "prompt_length": len(user_prompt),
                "response_length": len(result) if result else 0
            })
            return result or ""
        except Exception as e:
            latency = time.time() - start
            self.call_history.append({
                "timestamp": time.time(),
                "model": self.model,
                "latency": latency,
                "success": False,
                "error": str(e)
            })
            logger.warning(f"LLM调用失败: {e}")
            return ""

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        """
        调用LLM并解析JSON响应

        在user_prompt中要求LLM返回JSON格式。
        自动解析并返回dict，解析失败返回{}。
        """
        json_prompt = user_prompt + "\n\n请严格返回JSON格式，不要包含其他内容。"
        response = self.chat(system_prompt, json_prompt)
        if not response:
            return {}
        try:
            # 尝试提取JSON部分
            text = response.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except json.JSONDecodeError:
            logger.warning(f"JSON解析失败，返回原始文本")
            return {"raw_response": response}

    def analyze_experiment(self, experiment_history: List[Dict]) -> Dict[str, Any]:
        """
        分析实验历史，返回决策建议

        Returns:
            {"decision": "CONTINUE"|"PIVOT"|"STOP", "reason": "...", "suggested_config": {...}}
        """
        if not experiment_history:
            return {"decision": "CONTINUE", "reason": "首次实验", "suggested_config": {}}

        system = "你是一位机器学习实验分析专家。请分析实验历史并给出决策建议。"
        user = f"实验历史:\n{json.dumps(experiment_history[-5:], indent=2, ensure_ascii=False)}\n\n"
        user += "请分析并返回JSON: {\"decision\": \"CONTINUE/PIVOT/STOP\", \"reason\": \"...\", \"suggested_config\": {...}}"

        result = self.chat_json(system, user)
        if not result or "decision" not in result:
            return {"decision": "CONTINUE", "reason": "LLM未返回有效决策", "suggested_config": {}}
        return result

    def generate_code(self, task_description: str, existing_code: str = "",
                      constraints: str = "") -> str:
        """
        根据描述生成代码

        Args:
            task_description: 任务描述
            existing_code: 已有代码（修改场景）
            constraints: 约束条件

        Returns:
            生成的代码字符串
        """
        system = "你是一位PyTorch深度学习专家。请根据要求编写高质量、可运行的Python代码。"
        user = f"任务描述:\n{task_description}\n\n"
        if existing_code:
            user += f"现有代码:\n```python\n{existing_code[:3000]}\n```\n\n"
        if constraints:
            user += f"约束条件:\n{constraints}\n\n"
        user += "请只返回代码，不要包含解释。"

        return self.chat(system, user)

    def save_logs(self, path: str):
        """保存调用日志到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.call_history, f, ensure_ascii=False, indent=2)

    def _call_api(self, messages: List[Dict], temperature: float,
                  max_tokens: int) -> Optional[str]:
        """底层API调用"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
