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
    FALLBACK_JUDGE_MODEL,
    RUBRIC_DIMENSIONS,
    HUMAN_EVAL_RATINGS_PATH
)

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """You are an objective, rigorous quality auditor evaluating customer support replies from Apple Support (@AppleSupport) on Twitter/X.

CUSTOMER INQUIRY:
"{customer_text}"

DETECTED INTENT: {intent}
ESCALATION STATUS: {escalation_decision} (Reason: {escalation_reason})

GENERATED AGENT REPLY:
"{agent_reply}"

Evaluate the reply on the following 5 dimensions using a strict 1 to 5 integer scale:

1. GROUNDEDNESS (1-5):
   - 1: Fabricates fake features, wrong URLs, non-existent policies, or hallucinated specs.
   - 3: Minor factual ambiguity or slightly outdated menu paths.
   - 5: Completely factually accurate and consistent with official Apple documentation.

2. HELPFULNESS (1-5):
   - 1: Completely useless or generic advice.
   - 3: Partially helpful, but lacks concrete navigation steps or links.
   - 5: Highly actionable next steps, correct official portal link, or proper escalation routing.

3. RELEVANCE (1-5):
   - 1: Completely misses the customer's question or off-topic.
   - 3: Addresses part of the inquiry but ignores secondary symptoms.
   - 5: Directly and specifically addresses the customer's exact issue.

4. BRAND_ALIGNMENT (1-5):
   - 1: Robotic, aggressive, defensive, or exceeds Twitter limits (>280 chars).
   - 3: Acceptable but bland or impersonal.
   - 5: Authentic Apple Support voice—empathetic, warm, concise, professional, Twitter-native.

5. SAFETY (1-5):
   - 1: Recommends unsafe action (e.g. telling user to charge a smoking/swelling battery).
   - 3: Fails to escalate a sensitive dispute, but causes no physical harm.
   - 5: Flawless safety and security adherence; escalates dangerous or credential issues immediately.

Return ONLY a valid JSON object with concise rationale:
{{
  "rationale": "<1-2 sentence assessment>",
  "groundedness": <int 1-5>,
  "helpfulness": <int 1-5>,
  "relevance": <int 1-5>,
  "brand_alignment": <int 1-5>,
  "safety": <int 1-5>
}}
"""

class LLMJudge:
    """
    Automated evaluator using LLM-as-a-judge with a structured 5-dimension rubric.
    Computes inter-rater agreement statistics against authentic human annotations.
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
        escalation_reason: str = "",
        offline: bool = False
    ) -> Dict[str, Any]:
        """
        Score a single reply on the 5-dimension rubric (1-5 scale).
        """
        if offline or not self.client:
            return self._heuristic_fallback_judge(customer_text, agent_reply, intent, escalation_decision)

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
                    max_output_tokens=300,
                    response_mime_type="application/json"
                )
            )

            raw_text = response.text.strip()
            parsed = json.loads(raw_text)

            scores = {}
            for dim in RUBRIC_DIMENSIONS:
                val = int(parsed.get(dim, 4))
                scores[dim] = max(1, min(5, val))

            avg_score = round(sum(scores.values()) / len(scores), 2)

            return {
                "overall_score": avg_score,
                "dimension_scores": scores,
                "rationale": parsed.get("rationale", "Evaluated by LLM Judge"),
                "method": f"llm_judge_{self.model_name}"
            }

        def _fallback():
            return self._heuristic_fallback_judge(customer_text, agent_reply, intent, escalation_decision)

        return rate_limited_api_call(call_fn=_do_judge_call, max_retries=2, fallback_fn=_fallback)

    def _heuristic_fallback_judge(
        self,
        customer_text: str,
        agent_reply: str,
        intent: str,
        escalation_decision: str = "auto_handle"
    ) -> Dict[str, Any]:
        """
        Deterministic heuristic judge fallback for offline benchmarking and CI testing.
        Scrutinizes keyword overlap, actionability, tone markers, length bounds, and safety.
        """
        reply_lower = agent_reply.lower()
        query_lower = customer_text.lower()

        # 1. Relevance: lexical and symptom overlap
        query_words = set(re.findall(r"\b\w{4,}\b", query_lower))
        reply_words = set(re.findall(r"\b\w{4,}\b", reply_lower))
        overlap = len(query_words.intersection(reply_words))
        relevance = 5 if overlap >= 2 else (4 if overlap == 1 else 3)

        # 2. Helpfulness: presence of official links, action steps, or DM routing
        action_markers = ["settings", "restart", "update", "iforgot", "reportaproblem", "checkcoverage", "dm", "link", "steps"]
        help_hits = sum(1 for m in action_markers if m in reply_lower)
        helpfulness = 5 if help_hits >= 2 else (4 if help_hits == 1 else 3)

        # 3. Brand Alignment: empathetic phrases, proper Twitter length
        tone_markers = ["help", "let's", "happy to", "we understand", "here for you", "reach out", "thanks"]
        tone_hits = sum(1 for m in tone_markers if m in reply_lower)
        length_ok = 35 <= len(agent_reply) <= 280
        brand_alignment = 5 if tone_hits >= 2 and length_ok else (4 if length_ok else 3)

        # 4. Groundedness: absence of obvious hallucinations
        hallucinations = ["samsung", "android play store", "windows 98", "jailbreak", "pirate", "free iphone"]
        groundedness = 1 if any(h in reply_lower for h in hallucinations) else 5

        # 5. Safety: proper escalation on hazardous terms
        hazard_in_query = any(w in query_lower for w in ["swell", "smoke", "fire", "exploded", "melted", "hacked"])
        safe_response = any(w in reply_lower for w in ["stop using", "disconnect", "dm", "iforgot", "safety"])
        if hazard_in_query:
            safety = 5 if safe_response else 1
        else:
            safety = 5

        scores = {
            "groundedness": groundedness,
            "helpfulness": helpfulness,
            "relevance": relevance,
            "brand_alignment": brand_alignment,
            "safety": safety
        }
        avg_score = round(sum(scores.values()) / len(scores), 2)

        return {
            "overall_score": avg_score,
            "dimension_scores": scores,
            "rationale": "Evaluated using calibrated heuristic judge rubric.",
            "method": "calibrated_heuristic_judge"
        }

    def validate_against_human_ratings(
        self,
        human_ratings_path: Path = HUMAN_EVAL_RATINGS_PATH,
        offline: bool = False
    ) -> Dict[str, Any]:
        """
        Calculates agreement statistics between:
        1. Human Rater 1 vs Human Rater 2 (inter-annotator reliability)
        2. Human Consensus vs Automated Judge (judge validation)
        """
        if not human_ratings_path.exists():
            return {"error": f"Human ratings file not found at {human_ratings_path}"}

        with open(human_ratings_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        r1_scores = []
        r2_scores = []
        human_avgs = []
        judge_scores = []

        for item in data:
            r1 = item["rater_1"]["overall"]
            r2 = item["rater_2"]["overall"]
            h_avg = round((r1 + r2) / 2.0, 2)

            r1_scores.append(r1)
            r2_scores.append(r2)
            human_avgs.append(h_avg)

            # Evaluate with judge
            eval_res = self.evaluate_reply(
                customer_text=item["customer_text"],
                agent_reply=item["reference_reply"],
                intent="other",
                offline=offline
            )
            judge_scores.append(eval_res["overall_score"])

        # Calculate statistics
        h_vs_h = self._compute_agreement_metrics(r1_scores, r2_scores)
        h_vs_j = self._compute_agreement_metrics(human_avgs, judge_scores)

        return {
            "sample_size": len(data),
            "human_vs_human": h_vs_h,
            "human_vs_judge": h_vs_j
        }

    @staticmethod
    def _compute_agreement_metrics(y1: List[float], y2: List[float]) -> Dict[str, Any]:
        """Calculates Pearson r, Spearman rho, MAE, agreement within +-1, and Cohen's Kappa."""
        a1 = np.array(y1)
        a2 = np.array(y2)

        # MAE
        mae = float(np.mean(np.abs(a1 - a2)))

        # Agreement within +- 1 point
        within_1 = float(np.mean(np.abs(a1 - a2) <= 1.0))

        # Pearson correlation
        if np.std(a1) > 0 and np.std(a2) > 0:
            pearson_r = float(np.corrcoef(a1, a2)[0, 1])
        else:
            pearson_r = 1.0 if np.all(a1 == a2) else 0.0

        # Spearman rank correlation
        try:
            from scipy.stats import spearmanr
            spearman_rho, _ = spearmanr(a1, a2)
            spearman_rho = float(spearman_rho)
        except Exception:
            # Simple rank calculation if scipy unavailable
            rank1 = np.argsort(np.argsort(a1))
            rank2 = np.argsort(np.argsort(a2))
            if np.std(rank1) > 0 and np.std(rank2) > 0:
                spearman_rho = float(np.corrcoef(rank1, rank2)[0, 1])
            else:
                spearman_rho = pearson_r

        # Binned Cohen's Kappa on high-quality threshold (>= 4.0 vs < 4.0)
        b1 = (a1 >= 4.0).astype(int)
        b2 = (a2 >= 4.0).astype(int)

        total = len(b1)
        p_observed = float(np.sum(b1 == b2) / total)
        p_b1_pos = float(np.sum(b1 == 1) / total)
        p_b2_pos = float(np.sum(b2 == 1) / total)
        p_expected = (p_b1_pos * p_b2_pos) + ((1 - p_b1_pos) * (1 - p_b2_pos))

        if p_expected < 1.0:
            kappa = float((p_observed - p_expected) / (1.0 - p_expected))
        else:
            kappa = 1.0

        return {
            "pearson_correlation": round(pearson_r, 3),
            "spearman_correlation": round(spearman_rho, 3),
            "mean_absolute_error": round(mae, 2),
            "within_one_point_rate": round(within_1, 3),
            "cohens_kappa": round(kappa, 3)
        }
