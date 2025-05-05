import logging
from typing import List, Dict, Optional
from openai import AsyncOpenAI
import asyncio


class AsyncAIClient:
    _logger = logging.getLogger("AsyncAIClient")

    def __init__(self, api_key: str, base_url: str):
        self._logger.info("Initializing Async AI Client with base_url: %s", base_url)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.messages: List[Dict[str, str]] = []
        self._logger.debug("Async client initialized with %d initial messages", len(self.messages))

    def add_system_message(self, content: str) -> None:
        self._add_message("system", content)
        self._logger.debug("Added system message: %s", self._truncate_content(content))

    def add_user_message(self, content: str) -> None:
        self._add_message("user", content)
        self._logger.debug("Added user message: %s", self._truncate_content(content))

    def add_assistant_message(self, content: str) -> None:
        self._add_message("assistant", content)
        self._logger.debug("Added assistant message: %s", self._truncate_content(content))

    def _add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._logger.debug("Message added. Total messages: %d", len(self.messages))

    def _truncate_content(self, content: str) -> str:
        return content[:50] + "..." if len(content) > 50 else content

    def clear_messages(self) -> None:
        self.messages.clear()
        self._logger.info("All messages cleared")

    async def send_request(
        self,
        model: str = "openai/gpt-3.5-turbo-1106",
        temperature: float = 0.7,
        n: int = 1,
        max_tokens: int = 3000,
        extra_headers: Optional[Dict] = None
    ) -> str:
        self._logger.info(
            "Sending async request with params: model=%s, temperature=%s, n=%s, max_tokens=%s",
            model, temperature, n, max_tokens
        )
        headers = extra_headers or {"X-Title": "My App"}

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=self.messages,
                temperature=temperature,
                n=n,
                max_tokens=max_tokens,
                extra_headers=headers
            )
            self._logger.debug("Received successful async response")
            return response.choices[0].message.content
        except Exception as e:
            self._logger.error("Async request failed: %s", str(e), exc_info=True)
            raise


async def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    ai_client = AsyncAIClient(
        api_key="sk-or-vv-264b7ac948300c5bd342c7fe83339dd3b38a269668c36e4fbad3fca8ee859345",
        base_url="https://api.vsegpt.ru/v1"
    )

    try:
        ai_client.add_user_message("Напиши последовательно числа от 1 до 10")
        response = await ai_client.send_request()
        print("Async Response:", response)
    except Exception as e:
        logging.error("Async application error: %s", str(e))
    finally:
        ai_client.clear_messages()


if __name__ == "__main__":
    asyncio.run(main())