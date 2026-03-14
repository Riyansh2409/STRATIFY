"""
generator.py
============
RAG Generator — builds prompts for Llama-3 8B and runs inference.

Takes a RAGResponse from the retriever and:
  1. Builds a structured prompt (system + context + question)
  2. Runs Llama-3 8B inference (local HuggingFace pipeline)
  3. Returns a GeneratedAnswer with:
       - answer text (numbered, explained)
       - cited sources
       - confidence score
       - token usage

Prompt template follows Llama-3 Chat format:
  <|begin_of_text|>
  <|start_header_id|>system<|end_header_id|>
  {system_prompt}
  <|eot_id|>
  <|start_header_id|>user<|end_header_id|>
  {user_message}
  <|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from retriever import RAGResponse

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

LLAMA3_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

@dataclass
class GeneratorConfig:
    model_id:          str   = LLAMA3_MODEL_ID
    max_new_tokens:    int   = 512
    temperature:       float = 0.2       # low = factual, high = creative
    top_p:             float = 0.9
    repetition_penalty: float = 1.1
    load_in_4bit:      bool  = True       # QLoRA / bitsandbytes 4-bit
    device_map:        str   = "auto"    # "auto" | "cpu" | "cuda:0"

    # Prompt tuning
    cite_sources:      bool = True       # instruct model to cite [Context N]
    numbered_output:   bool = True       # instruct model to use 1. 2. 3. format
    explain_reasoning: bool = True       # instruct model to explain each point


# ─────────────────────────────────────────────────────────────
# Answer dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class GeneratedAnswer:
    query:          str
    answer:         str
    sources:        list[dict]         # from RAGResponse.sources
    model_id:       str
    tokens_used:    int = 0
    latency_sec:    float = 0.0
    confidence:     float = 0.0        # mean cross-encoder score of top passages
    retrieval_stats: dict = field(default_factory=dict)

    def print_answer(self):
        """Pretty-print to console."""
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"Query:  {self.query}")
        print(sep)
        print(self.answer)
        print(sep)
        print("Sources:")
        for s in self.sources:
            print(f"  [{s['index']}] {s['file']}  (score={s['final_score']:.3f})")
        print(f"\nTokens: {self.tokens_used}  |  Latency: {self.latency_sec:.2f}s")
        print(sep + "\n")

    def to_dict(self) -> dict:
        return {
            "query":          self.query,
            "answer":         self.answer,
            "sources":        self.sources,
            "model_id":       self.model_id,
            "tokens_used":    self.tokens_used,
            "latency_sec":    round(self.latency_sec, 3),
            "confidence":     round(self.confidence, 4),
            "retrieval_stats": self.retrieval_stats,
        }


# ─────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────

def build_llama3_prompt(
    query: str,
    context: str,
    config: GeneratorConfig,
) -> str:
    """
    Build a Llama-3 Instruct chat-format prompt.

    The system message instructs the model to:
      - Answer only from the provided context
      - Cite sources as [Context N]
      - Use numbered list format
      - Explain reasoning clearly
    """
    rules = []
    if config.cite_sources:
        rules.append("- Cite every fact using [Context N] notation (e.g. [Context 1]).")
    if config.numbered_output:
        rules.append("- Structure your answer as a numbered list (1. 2. 3. ...).")
    if config.explain_reasoning:
        rules.append("- After each point, briefly explain why it matters.")
    rules.append("- If the context does not contain enough information, say so clearly.")
    rules.append("- Do NOT make up information not present in the context.")

    system_prompt = (
        "You are an expert AI assistant specialising in machine learning and data analysis. "
        "Answer questions using ONLY the provided context passages. "
        "Follow these rules:\n" + "\n".join(rules)
    )

    user_message = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Please provide a comprehensive, numbered answer with citations."
    )

    # Llama-3 Instruct chat template
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_message}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt


# ─────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────

class RAGGenerator:
    """
    Runs Llama-3 8B inference on retrieved context.

    Supports:
      - Full precision (float16)
      - 4-bit quantisation (bitsandbytes, default)
      - CPU fallback for testing

    Usage:
        gen = RAGGenerator(config=GeneratorConfig())
        answer = gen.generate(rag_response)
        answer.print_answer()
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self._pipeline = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
                pipeline,
            )

            logger.info(f"Loading Llama-3 model: {self.config.model_id}")

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                use_fast=True,
            )
            tokenizer.pad_token = tokenizer.eos_token

            model_kwargs: dict = {
                "torch_dtype": torch.float16,
                "device_map": self.config.device_map,
            }
            if self.config.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["quantization_config"] = bnb_config

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs,
            )

            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            self._tokenizer = tokenizer
            logger.info("Llama-3 pipeline ready.")

        except Exception as e:
            logger.warning(
                f"Could not load Llama-3 ({e}). "
                "Running in stub mode — answers will be template strings (for testing)."
            )
            self._pipeline   = None
            self._tokenizer  = None

    def generate(self, rag_response: RAGResponse) -> GeneratedAnswer:
        """
        Generate an answer from a RAGResponse.

        Args:
            rag_response: Output from RAGRetriever.retrieve()

        Returns:
            GeneratedAnswer with answer text, sources, and metadata
        """
        if not rag_response.has_results():
            return GeneratedAnswer(
                query=rag_response.query,
                answer="No relevant context found. Please rephrase or provide more documents.",
                sources=[],
                model_id=self.config.model_id,
            )

        prompt = build_llama3_prompt(
            rag_response.query,
            rag_response.context,
            self.config,
        )

        start = time.time()

        if self._pipeline is None:
            # Stub mode (no GPU / model not downloaded)
            answer = self._stub_answer(rag_response)
            tokens_used = len(prompt.split())
        else:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=self.config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                return_full_text=False,
            )
            answer = outputs[0]["generated_text"].strip()
            tokens_used = len(self._tokenizer.encode(prompt + answer))

        latency = time.time() - start

        # Confidence = mean cross-encoder score of top passages
        scores = [p.cross_score for p in rag_response.passages]
        confidence = sum(scores) / len(scores) if scores else 0.0

        return GeneratedAnswer(
            query=rag_response.query,
            answer=answer,
            sources=rag_response.sources,
            model_id=self.config.model_id,
            tokens_used=tokens_used,
            latency_sec=latency,
            confidence=confidence,
            retrieval_stats=rag_response.retrieval_stats,
        )

    def _stub_answer(self, rag: RAGResponse) -> str:
        """Deterministic stub answer (used when model is not loaded)."""
        source_refs = ", ".join(
            f"[Context {s['index']}]" for s in rag.sources[:3]
        )
        snippet = rag.passages[0].snippet(120) if rag.passages else ""
        return (
            f"1. Based on the retrieved context {source_refs}, here is what I found:\n\n"
            f"   {snippet}\n\n"
            "2. This is a stub response — Llama-3 model is not loaded in this environment.\n"
            "   To get real answers, download the model weights and set load_in_4bit=True.\n\n"
            "3. Retrieved sources are listed below with their relevance scores."
        )
