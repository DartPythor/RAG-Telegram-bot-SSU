import os
import asyncio
import logging
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter
import aiofiles

logging.basicConfig(level=logging.INFO)


async def generate_documents() -> List[Document]:
    async with aiofiles.open("data/data.md", encoding="utf-8") as f:
        programs_text = await f.read()

    headers_to_split_on = [
        ("#", "Action"),
        ("##", "SubAction")
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = await asyncio.to_thread(splitter.split_text, programs_text)

    documents_list = []
    for chunk in chunks:
        doc_content = (
            f"Событие: {chunk.metadata['Action']}\n"
            f"Под событие: {chunk.metadata['SubAction']}\n"
            f"Информация: {chunk.page_content}"
        )

        documents_list.append(
            Document(
                page_content=doc_content.lower(),
                metadata=chunk.metadata
            )
        )

    return documents_list


class AsyncEmbeddingFactory:
    EMBEDDINGS = {
        "huggingface": HuggingFaceEmbeddings,
    }

    @staticmethod
    async def create_embedding(embedding_type="huggingface", **kwargs):
        if embedding_type in AsyncEmbeddingFactory.EMBEDDINGS:
            return await asyncio.to_thread(
                AsyncEmbeddingFactory.EMBEDDINGS[embedding_type], **kwargs
            )
        raise ValueError(f"Unknown type for embedding, choose form {AsyncEmbeddingFactory.EMBEDDINGS}")


class AsyncFAISSManager:
    def __init__(self, vector_store_name, embedding_type, embedding_kwargs):
        self.embeddings = None
        self.vector_store_directory = "vector_store"
        self.vector_store_name = vector_store_name
        self.vector_store = None
        self.embedding_type = embedding_type
        self.embedding_kwargs = embedding_kwargs

    async def initialize(self):
        self.embeddings = await AsyncEmbeddingFactory.create_embedding(
            self.embedding_type,
            **self.embedding_kwargs,
        )

    async def load_vector_store(self):
        try:
            self.vector_store = await asyncio.to_thread(
                FAISS.load_local,
                folder_path=self.vector_store_directory,
                index_name=self.vector_store_name,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            self.vector_store = None
            logging.error(f"Ошибка загрузки базы FAISS: {str(e)}")

    async def generate_vector_store(self, documents: List[Document]):
        logging.info("############ GENERATE_VECTOR_STORE ############")
        self.vector_store = await asyncio.to_thread(
            FAISS.from_documents,
            documents=documents,
            embedding=self.embeddings,
        )
        await asyncio.to_thread(
            self.vector_store.save_local,
            folder_path=self.vector_store_directory,
            index_name=self.vector_store_name
        )

    async def add_documents(self, documents: List[Document]):
        if self.vector_store is None:
            await self.generate_vector_store(documents)
        else:
            await asyncio.to_thread(
                self.vector_store.add_documents,
                documents
            )

    async def similarity_search(self, question, meta_filter=None, k=1):
        """Асинхронный поиск похожих документов"""
        logging.info("############ SIMILARITY_SEARCH ############")
        logging.info(f"Вопрос: {question} Фильтр: {meta_filter} k={k}")

        return await asyncio.to_thread(
            self.vector_store.similarity_search,
            query=question.lower(),
            k=k,
            filter=meta_filter
        )

    async def similarity_search_with_score(self, question, meta_filter=None, k=1):
        """Асинхронный поиск с оценкой"""
        logging.info("############ SIMILARITY_SEARCH_WITH_SCORE ############")
        logging.info(f"Вопрос: {question} Фильтр: {meta_filter} K={k}")

        return await asyncio.to_thread(
            self.vector_store.similarity_search_with_score,
            query=question.lower(),
            k=k,
            filter=meta_filter,
        )

    async def search_retriever(self, question, meta_filter=None, k=1, search_type="similarity"):
        """Асинхронный поиск через retriever"""
        logging.info("############ RETRIEVER ############")
        logging.info(f"Вопрос: {question} Фильтр: {meta_filter} K={k} search_type={search_type}")

        retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k, "filter": meta_filter},
        )

        return await asyncio.to_thread(
            retriever.invoke,
            question.lower()
        )


async def main():
    faiss_manager = AsyncFAISSManager(
        vector_store_name="rag",
        embedding_type="huggingface",
        embedding_kwargs={
            "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "model_kwargs": {"trust_remote_code": True, "device": "cuda"},
        },
    )
    await faiss_manager.initialize()

    documents = await generate_documents()
    await faiss_manager.generate_vector_store(documents)

    results = await faiss_manager.similarity_search_with_score("Стипендия", k=3)
    for doc, score in results:
        logging.info(f"[Score: {score:.3f}] {doc.page_content} | Metadata: {doc.metadata}")


if __name__ == "__main__":
    asyncio.run(main())