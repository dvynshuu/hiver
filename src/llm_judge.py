import re
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    GEMINI_API_KEY,
    JUDGE_MODEL_NAME,
    FALLBACK_JUDGE_MODEL
)

logger = logging.getLogger(__name__)

# Rubric definitions
RUBRIC_DIMENSIONS = [
    "relevance",
    "helpfulness",
    "tone",
    "groundedness",
    "completeness"
]

JUDGE_PROMPT_TEMPLATE = """You are an impartial, highly rigorous quality auditor evaluating customer support replies from Apple Support (@AppleSupport) on Twitter.

CUSTOMER INQUIRY:
"{customer_text}"

DETECTED INTENT: {intent}
ESCALATION DECISION: {escalation_decision} (Reason: {escalation_reason})

GENERATED AGENT REPLY:
"{agent_reply}"

Evaluate the agent's reply on the following 5 dimensions using a strict 1 to 5 scale:

1. RELEVANCE (1-5): Does the reply directly address the customer's specific issue?
   - 1: Off-topic or ignores the query completely.
   - 3: Partially relevant but misses the core symptom.
   - 5: Perfectly targeted at the customer's exact issue.

2. HELPFULNESS (1-5): Does the reply provide actionable steps, official tools, or appropriate DM links?
   - 1: Unhelpful, vague, or useless advice.
   - 3: Generic troubleshooting (e.g. just 'restart') without specific steps.
   - 5: Highly practical, clear, actionable next steps or proper routing.

3. TONE (1-5): Does it sound like Apple's brand voice? (Empathetic, polite, professional, concise, Twitter-friendly <280 chars)
   - 1: Rude, robotic, defensive, or overly wordy.
   - 3: Acceptable but bland or slightly awkward.
   - 5: Authentic Apple Support voice—warm, empathetic, concise, and professional.

4. GROUNDEDNESS (1-5): Is the information factually accurate to the Apple ecosystem? (No hallucinated products, fake settings, or misleading policies)
   - 1: Fabricates fake features, wrong menus, or false promises.
   - 3: Minor factual inaccuracies or slightly outdated menu paths.
   - 5: 100% factually accurate, realistic troubleshooting steps.

5. COMPLETENESS (1-5): Does the response address all questions or hand off appropriately if escalation is needed?
   - 1: Completely incomplete.
   - 3: Addresses part of the inquiry, leaves other questions unanswered.
   - 5: Thoroughly addresses the entire inquiry or provides complete next steps.

IMPORTANT: Write your brief Chain-of-Thought critique first, then assign integer scores (1-5) for each dimension.

Return ONLY a valid JSON object matching this schema:
{{
  "critique": "<2-3 sentence honest assessment>",
  "relevance": <integer 1-5>,
  "helpfulness": <integer 1-5>,
  "tone": <integer 1-5>,
  "groundedness": <integer 1-5>,
  "completeness": <integer 1-5>
}}
"""

class LLMJudge:
    """
    Automated evaluator using LLM-as-a-judge with structured multi-dimensional rubric.
    Computes inter-rater agreement against human annotations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = JUDGE_MODEL_NAME
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client for LLM Judge: {e}")

    def evaluate_reply(
        self,
        customer_text: str,
        agent_reply: str,
        intent: str = "other",
        escalation_decision: str = "auto_handle",
        escalation_reason: str = ""
    ) -> Dict[str, Any]:
        """
        Score a single customer reply using the 5-dimension rubric.
        """
        if not self.client:
            return self._heuristic_fallback_judge(customer_text, agent_reply, intent)

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            customer_text=customer_text,
            intent=intent,
            escalation_decision=escalation_decision,
            escalation_reason=escalation_reason,
            agent_reply=agent_reply
        )

        from src.rate_limiter import rate_limited_api_call

        def _do_judge_call():
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1000,
                    response_mime_type="application/json"
                )
            )

            raw_text = response.text.strip()
            parsed = json.loads(raw_text)

            # Ensure scores are clamped between 1 and 5
            scores = {}
            for dim in RUBRIC_DIMENSIONS:
                val = int(parsed.get(dim, 4))
                scores[dim] = max(1, min(5, val))

            avg_score = round(sum(scores.values()) / len(scores), 2)

            return {
                "overall_score": avg_score,
                "dimension_scores": scores,
                "critique": parsed.get("critique", "Evaluated by LLM Judge"),
                "method": f"llm_judge_{self.model_name}"
            }

        def _fallback():
            return self._heuristic_fallback_judge(customer_text, agent_reply, intent)

        return rate_limited_api_call(
            call_fn=_do_judge_call,
            max_retries=3,
            fallback_fn=_fallback
        )

    def _heuristic_fallback_judge(
        self,
        customer_text: str,
        agent_reply: str,
        intent: str
    ) -> Dict[str, Any]:
        """
        Calibrated rule-based judge fallback when API is offline.
        Uses linguistic markers, length penalties, and domain alignment.
        """
        reply_lower = agent_reply.lower()
        query_lower = customer_text.lower()

        # Relevance: check keyword overlap
        query_words = set(re.findall(r"\w{4,}", query_lower))
        reply_words = set(re.findall(r"\w{4,}", reply_lower))
        overlap = len(query_words.intersection(reply_words))
        relevance = 5 if overlap >= 2 else (4 if overlap == 1 else 3)

        # Helpfulness: check for actionable advice
        action_markers = ["step", "setting", "restart", "update", "dm", "link", "visit", "apple.com", "try"]
        help_hits = sum(1 for m in action_markers if m in reply_lower)
        helpfulness = 5 if help_hits >= 3 else (4 if help_hits >= 1 else 3)

        # Tone: empathetic phrases vs defensive
        tone_markers = ["help", "let's", "happy to", "we understand", "here for you", "reach out", "thanks"]
        tone_hits = sum(1 for m in tone_markers if m in reply_lower)
        # Length penalty: twitter replies shouldn't be 1000 chars
        length_ok = 40 <= len(agent_reply) <= 380
        tone = 5 if tone_hits >= 2 and length_ok else (4 if length_ok else 3)

        # Groundedness: check for hallucination keywords
        hallucination_markers = ["samsung", "android play store", "windows 98", "jailbreak", "pirate"]
        has_hallucination = any(h in reply_lower for h in hallucination_markers)
        groundedness = 1 if has_hallucination else 5

        # Completeness: DM offer or direct solution
        has_closure = any(c in reply_lower for c in ["dm", "let us know", "assist", "help", "support"])
        completeness = 4 if has_closure else 3

        scores = {
            "relevance": relevance,
            "helpfulness": helpfulness,
            "tone": tone,
            "groundedness": groundedness,
            "completeness": completeness
        }
        avg_score = round(sum(scores.values()) / len(scores), 2)

        return {
            "overall_score": avg_score,
            "dimension_scores": scores,
            "critique": "Evaluated using calibrated heuristic judge rubric.",
            "method": "calibrated_heuristic_judge"
        }

    @staticmethod
    def calculate_human_agreement(
        human_ratings: List[float],
        judge_ratings: List[float]
    ) -> Dict[str, Any]:
        """
        Calculates Pearson Correlation (r) and Binned Cohen's Kappa between Human and LLM Judge ratings.
        Binned Kappa evaluates agreement on whether an answer is 'High Quality' (>= 4.0) vs 'Needs Improvement' (< 4.0).
        """
        if len(human_ratings) != len(judge_ratings) or len(human_ratings) == 0:
            return {"error": "Mismatched or empty rating arrays"}

        h = np.array(human_ratings)
        j = np.array(judge_ratings)

        # Mean Absolute Error
        mae = float(np.mean(np.abs(h - j)))

        # Pearson correlation
        if np.std(h) > 0 and np.std(j) > 0:
            corr = float(np.corrcoef(h, j)[0, 1])
        else:
            corr = 1.0 if np.all(h == j) else 0.0

        # Cohen's Kappa on binned binary quality (>= 4.0 vs < 4.0)
        h_bin = (h >= 4.0).astype(int)
        j_bin = (j >= 4.0).astype(int)

        # Confusion matrix for binary agreement
        total = len(h_bin)
        agree = np.sum(h_bin == j_bin)
        p_observed = agree / total

        p_h_pos = np.sum(h_bin == 1) / total
        p_j_pos = np.sum(j_bin == 1) / total
        p_expected = (p_h_pos * p_j_pos) + ((1 - p_h_pos) * (1 - p_j_pos))

        if p_expected < 1.0:
            kappa = float((p_observed - p_expected) / (1.0 - p_expected))
        else:
            kappa = 1.0

        return {
            "sample_size": total,
            "pearson_correlation": round(corr, 3),
            "mean_absolute_error": round(mae, 2),
            "observed_agreement_rate": round(p_observed, 3),
            "cohens_kappa": round(kappa, 3),
            "agreement_strength": "Substantial" if kappa >= 0.60 else ("Moderate" if kappa >= 0.40 else "Fair")
        }
