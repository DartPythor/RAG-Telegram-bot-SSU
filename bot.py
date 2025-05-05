import asyncio
import os
import logging
from typing import Dict
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message

from aiclient import AsyncAIClient
from fais import AsyncFAISSManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatelessAIBot:
    def __init__(self):
        self.bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
        self.dp = Dispatcher()
        self.faiss_manager = AsyncFAISSManager(
            vector_store_name="rag",
            embedding_type="huggingface",
            embedding_kwargs={
                "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "model_kwargs": {"trust_remote_code": True, "device": "cuda"},
            }
        )

        self.dp.message(Command(commands=["start"]))(self.start_handler)
        self.dp.message()(self.message_handler)

    async def initialize(self):
        await self.faiss_manager.initialize()
        await self.faiss_manager.load_vector_store()

    async def _get_context(self, question: str) -> str:
        try:
            docs = await self.faiss_manager.similarity_search(question, k=2)
            context = "\n".join([doc.page_content for doc in docs])
            logger.info(f"Найденный контекст: {context}")
            return context
        except Exception as e:
            logger.error(f"Ошибка поиска контекста: {str(e)}")
            return ""

    async def start_handler(self, message: Message) -> None:
        await message.answer(
            "Привет! Я AI-бот с доступом к базе знаний университета СГУ.\n"
            "Задайте мне вопрос, и я постараюсь найти ответ в базе знаний."
        )

    async def message_handler(self, message: Message) -> None:
        try:
            # Получаем контекст из векторного хранилища
            context = await self._get_context(message.text)

            # Создаем временного клиента для запроса
            client = AsyncAIClient(
                api_key=os.getenv("AI_API_KEY"),
                base_url=os.getenv("AI_BASE_URL")
            )

            # Формируем системное сообщение с контекстом
            system_message = (
                "Ты полезный ассистент СГУ. Отвечай только на основе предоставленного контекста. "
                f"Контекст: {context}\n\nВопрос: {message.text}"
            )

            client.add_system_message(system_message)
            client.add_user_message(message.text)

            # Отправляем запрос
            await self.bot.send_chat_action(message.chat.id, "typing")
            response = await client.send_request()

            await message.answer(response)

        except Exception as e:
            logger.error(f"Ошибка: {str(e)}")
            await message.answer("Произошла ошибка при обработке запроса. Попробуйте позже.")

    async def run(self):
        await self.initialize()
        await self.dp.start_polling(self.bot)


if __name__ == "__main__":
    load_dotenv()

    required_env = ["TELEGRAM_BOT_TOKEN", "AI_API_KEY", "AI_BASE_URL"]
    for var in required_env:
        if not os.getenv(var):
            raise ValueError(f"Необходимо установить переменную окружения {var}")

    bot = StatelessAIBot()
    asyncio.run(bot.run())
