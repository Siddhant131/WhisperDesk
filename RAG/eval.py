import os
import json
from dataclasses import dataclass
from openai import OpenAI

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3")

FAITHFULNESS_PROMPT = """You are an evaluation judge. Given a context and a generated answer, assess whether the answer is faithful to the context (no hallucinations or unsupported claims).

Context:
{context}

Answer:
{answer}

Respond ONLY with valid JSON:
{{
  "score": <integer 1-5>,
  "reason": "<one sentence>"
}}
Score 5 = fully grounded in context, 1 = mostly hallucinated."""

RELEVANCE_PROMPT = """You are an evaluation judge. Given a customer query and a generated answer, assess how relevant and helpful the answer is.

Query: {query}
Answer: {answer}

Respond ONLY with valid JSON:
{{
  "score": <integer 1-5>,
  "reason": "<one sentence>"
}}
Score 5 = directly addresses the query, 1 = completely off-topic."""

COMPLETENESS_PROMPT = """You are an evaluation judge. Given a customer query and a generated answer, assess whether the answer is complete and resolves the query.

Query: {query}
Answer: {answer}

Respond ONLY with valid JSON:
{{
  "score": <integer 1-5>,
  "reason": "<one sentence>"
}}
Score 5 = fully resolves the query, 1 = leaves query entirely unresolved."""


@dataclass
class EvalResult:
    faithfulness: float
    relevance: float
    completeness: float
    faithfulness_reason: str
    relevance_reason: str
    completeness_reason: str

    @property
    def aggregate(self) -> float:
        return round((self.faithfulness + self.relevance + self.completeness) / 3, 2)

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "aggregate": self.aggregate,
            "reasons": {
                "faithfulness": self.faithfulness_reason,
                "relevance": self.relevance_reason,
                "completeness": self.completeness_reason,
            },
        }


class LLMJudge:
    def __init__(self):
        self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    def _call_judge(self, prompt: str, retries: int = 3) -> dict:
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150,
                )
                raw = response.choices[0].message.content.strip()

                # Strip markdown code fences llama3 sometimes wraps around JSON
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()

                return json.loads(raw)

            except (json.JSONDecodeError, IndexError) as e:
                print(f"  [warn] Judge attempt {attempt + 1}/{retries} failed: {e}")
                if attempt == retries - 1:
                    print("  [warn] Using fallback score 3.")
                    return {"score": 3, "reason": "Judge could not produce a parseable response."}

        return {"score": 3, "reason": "Judge did not respond."}

    def evaluate(self, query: str, answer: str, context: str) -> EvalResult:
        faithfulness_result = self._call_judge(
            FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        )
        relevance_result = self._call_judge(
            RELEVANCE_PROMPT.format(query=query, answer=answer)
        )
        completeness_result = self._call_judge(
            COMPLETENESS_PROMPT.format(query=query, answer=answer)
        )

        return EvalResult(
            faithfulness=faithfulness_result["score"],
            relevance=relevance_result["score"],
            completeness=completeness_result["score"],
            faithfulness_reason=faithfulness_result["reason"],
            relevance_reason=relevance_result["reason"],
            completeness_reason=completeness_result["reason"],
        )


def run_eval_suite(
    pipeline,
    test_cases: list[dict],
    output_path: str = "eval_results.json",
) -> list[dict]:
    judge = LLMJudge()
    results = []

    for i, case in enumerate(test_cases):
        query = case["query"]
        print(f"[{i+1}/{len(test_cases)}] Evaluating: {query[:60]}...")

        from rag_pipeline import ConversationHistory
        history = ConversationHistory()
        output = pipeline.generate(query, history)

        eval_result = judge.evaluate(
            query=query,
            answer=output["answer"],
            context=output["context_used"],
        )

        record = {
            "query": query,
            "answer": output["answer"],
            "context": output["context_used"],
            "eval": eval_result.to_dict(),
        }

        if "expected" in case:
            record["expected"] = case["expected"]

        results.append(record)
        print(f"  → Aggregate score: {eval_result.aggregate}/5")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    scores = [r["eval"]["aggregate"] for r in results]
    avg = round(sum(scores) / len(scores), 2)
    print(f"\nEval complete. Average aggregate score: {avg}/5 → {output_path}")

    return results


if __name__ == "__main__":
    from rag_pipeline import RAGPipeline, ConversationHistory

    pipeline = RAGPipeline()

    test_cases = [
        {"query": "What is your return policy?"},
        {"query": "How do I track my order?"},
        {"query": "Can I change my delivery address?"},
    ]

    run_eval_suite(pipeline, test_cases)