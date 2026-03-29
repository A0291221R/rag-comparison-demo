"""
core/guardrails/validators.py — Input and output guardrails.

Input:  prompt injection detection, PII detection, query length
Output: hallucination grounding check, citation verification, toxicity
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class GuardrailResult(str, Enum):
    PASS = "pass"
    BLOCK = "block"
    WARN = "warn"


@dataclass
class ValidationResult:
    result: GuardrailResult
    reason: str = ""
    sanitized_text: str = ""
    score: float = 1.0


# ── Input Guardrails ──────────────────────────────────────────────────────────

class InputGuardrail:
    """
    Validates user queries before they hit the RAG pipeline.
    Blocks: prompt injection, excessive length, PII (optionally redacts).
    """

    # Common prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(everything|all)",
        r"you\s+are\s+now\s+",
        r"act\s+as\s+(if\s+you\s+are|a\s+)",
        r"jailbreak",
        r"DAN\s+mode",
        r"bypass\s+(your\s+)?(safety|restrictions|guidelines)",
        r"<\s*system\s*>",
        r"\[INST\]|\[/INST\]",
        r"###\s*System:",
    ]

    # PII patterns (redact, don't block)
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(\+?1?\s?)?(\(?\d{3}\)?[\s.\-]?)(\d{3}[\s.\-]?\d{4})\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
    }

    MAX_QUERY_LENGTH = 2000

    def __init__(self) -> None:
        self._compiled_injection = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self._compiled_pii = {k: re.compile(v) for k, v in self.PII_PATTERNS.items()}

    def validate(self, query: str) -> ValidationResult:
        # Length check
        if len(query) > self.MAX_QUERY_LENGTH:
            return ValidationResult(
                result=GuardrailResult.BLOCK,
                reason=f"Query exceeds maximum length ({len(query)} > {self.MAX_QUERY_LENGTH})",
                sanitized_text="",
            )

        # Injection detection
        for pattern in self._compiled_injection:
            if pattern.search(query):
                logger.warning("prompt_injection_detected", query=query[:100])
                return ValidationResult(
                    result=GuardrailResult.BLOCK,
                    reason="Potential prompt injection detected.",
                    sanitized_text="",
                )

        # PII redaction (warn, don't block)
        sanitized = query
        pii_found = []
        for pii_type, pattern in self._compiled_pii.items():
            if pattern.search(sanitized):
                pii_found.append(pii_type)
                sanitized = pattern.sub(f"[{pii_type.upper()}_REDACTED]", sanitized)

        if pii_found:
            logger.warning("pii_redacted", types=pii_found)
            return ValidationResult(
                result=GuardrailResult.WARN,
                reason=f"PII detected and redacted: {', '.join(pii_found)}",
                sanitized_text=sanitized,
                score=0.8,
            )

        return ValidationResult(
            result=GuardrailResult.PASS,
            sanitized_text=query,
            score=1.0,
        )


# ── Output Guardrails ─────────────────────────────────────────────────────────

class OutputGuardrail:
    """
    Validates generated answers before returning to the user.
    Checks: citation grounding, hallucination score, toxicity (basic).
    """

    MIN_GROUNDING_SCORE = 0.5

    TOXIC_PATTERNS = [
        r"\b(kill|murder|bomb|attack|hack|exploit)\b.{0,30}\b(person|people|human|system|server)\b",
    ]

    def __init__(self, llm: Any | None = None):
        self._llm = llm  # optional LLM for deep hallucination check
        self._compiled_toxic = [re.compile(p, re.IGNORECASE) for p in self.TOXIC_PATTERNS]

    def _citation_grounding_score(self, answer: str, context_chunks: list[str]) -> float:
        """
        Simple n-gram overlap between answer and retrieved context.
        In production, replace with NLI-based entailment model.
        """
        if not context_chunks:
            return 0.0
        answer_tokens = set(answer.lower().split())
        context_tokens = set(" ".join(context_chunks).lower().split())
        overlap = len(answer_tokens & context_tokens)
        score = overlap / max(len(answer_tokens), 1)
        return min(score, 1.0)

    def validate(
        self,
        answer: str,
        context_chunks: list[str],
        trace_id: str = "",
    ) -> ValidationResult:
        # Empty answer
        if not answer or len(answer.strip()) < 10:
            return ValidationResult(
                result=GuardrailResult.BLOCK,
                reason="Empty or too-short answer generated.",
                sanitized_text="",
                score=0.0,
            )

        # Toxicity check
        for pattern in self._compiled_toxic:
            if pattern.search(answer):
                logger.warning("toxic_output_detected", trace_id=trace_id)
                return ValidationResult(
                    result=GuardrailResult.BLOCK,
                    reason="Answer flagged for potentially harmful content.",
                    sanitized_text="",
                    score=0.0,
                )

        # Citation grounding
        grounding = self._citation_grounding_score(answer, context_chunks)
        if grounding < self.MIN_GROUNDING_SCORE:
            logger.warning(
                "low_grounding_score",
                score=grounding,
                trace_id=trace_id,
            )
            return ValidationResult(
                result=GuardrailResult.WARN,
                reason=f"Low citation grounding score: {grounding:.2f}",
                sanitized_text=answer,
                score=grounding,
            )

        return ValidationResult(
            result=GuardrailResult.PASS,
            sanitized_text=answer,
            score=grounding,
        )
