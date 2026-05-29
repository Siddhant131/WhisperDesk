import os
from dataclasses import dataclass, field
from openai import OpenAI
from retriever import RAGRetriever
from tts import speak_streaming

SYSTEM_PROMPT = """You are a professional customer support agent. Answer the customer's query clearly and concisely using only the provided context. If the context does not contain enough information, say so politely and offer to escalate the call."""

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

GENERATION_CONFIG = {
    "model": os.getenv("LLM_MODEL", "llama3"),
    "temperature": 0.3,
    "max_tokens": 300,
}


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ConversationHistory:
    messages: list[Message] = field(default_factory=list)
    max_turns: int = 10

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-(self.max_turns * 2):]

    def to_openai_format(self, system_prompt: str) -> list[dict]:
        history = [{"role": "system", "content": system_prompt}]
        for msg in self.messages:
            history.append({"role": msg.role, "content": msg.content})
        return history


class RAGPipeline:
    def __init__(self, kb_dir: str = "./knowledge_base", top_k: int = 5):
        self.retriever = RAGRetriever(kb_dir=kb_dir)
        self.top_k = top_k
        self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    def generate(self, query: str, history: ConversationHistory) -> dict:
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        context = self.retriever.format_context(retrieved)

        augmented_user_message = (
            f"Customer query: {query}\n\n"
            f"Relevant context:\n{context}"
        )

        history.add("user", query)

        messages = history.to_openai_format(SYSTEM_PROMPT)
        messages[-1]["content"] = augmented_user_message

        response = self.client.chat.completions.create(
            messages=messages,
            **GENERATION_CONFIG,
        )

        answer = response.choices[0].message.content.strip()
        history.add("assistant", answer)
        speak_streaming(answer)

        return {
            "answer": answer,
            "retrieved_chunks": retrieved,
            "context_used": context,
        }

    def __call__(self, query: str, history: ConversationHistory) -> str:
        result = self.generate(query, history)
        return result["answer"]
