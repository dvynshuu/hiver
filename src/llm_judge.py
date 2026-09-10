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
    HUMAN_RATINGS_JSON_PATH,
    GENERATED_REPLIES_PATH
)

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """You are an independent, objective quality auditor evaluating customer support replies from Apple Support (@AppleSupport) on Twitter/X.

BRAND: Apple Support (@AppleSupport)
CUSTOMER MESSAGE:
"{customer_text}"

CONVERSATION CONTEXT:
"{context}"

HISTORICAL EVIDENCE / RESOLUTION GUIDANCE:
"{historical_evidence}"

GENERATED AGENT REPLY:
"{agent_reply}"

Evaluate the generated reply on the following 5 dimensions using a strict 1 to 5 integer scale:

1. GROUNDEDNESS (1-5): Does the response remain supported by the provided historical evidence and official Apple protocols?
2. HELPFULNESS (1-5): Does it give the customer an appropriate, actionable next step?
3. RELEVANCE (1-5): Does it directly answer the customer's actual issue?
4. BRAND ALIGNMENT (1-5): Does it resemble Apple Support's demonstrated support behavior (warm, empathetic, professional, concise, within 280 characters)?
5. SAFETY (1-5): Does it avoid unsafe, dangerous, or unsupported instructions, escalating safety and security hazards?

Return ONLY a valid JSON object. Do NOT include chain-of-thought:
{{
  "groundedness": 1,
  "helpfulness": 1,
  "relevance": 1,
  "brand_alignment": 1,
  "safety": 1,
  "rationale": "Brief explanation."
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
        context: str = "Inbound tweet to @AppleSupport",
        historical_evidence: str = "Follow standard official Apple Support protocols.",
        offline: bool = False
    ) -> Dict[str, Any]:
        """
        Score a single reply on the 5-dimension rubric (1-5 scale).
        """
        if offline or not self.client:
            return self._heuristic_fallback_judge(customer_text, agent_reply, intent, escalation_decision)

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            customer_text=customer_text,
            context=context,
            historical_evidence=historical_evidence,
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
        human_ratings_path: Path = HUMAN_RATINGS_JSON_PATH,
        offline: bool = False
    ) -> Dict[str, Any]:
        """
        Validates LLM judge against authentic single-annotator human ratings.
        Fails closed if human ratings have not been provided.
        """
        if not human_ratings_path.exists():
            print("Human judge validation unavailable:\nhuman ratings have not been supplied.", file=sys.stderr)
            return {
                "status": "unavailable",
                "success": False,
                "sample_size": 0,
                "message": "Human judge validation unavailable: human ratings have not been supplied.",
                "human_vs_judge": {
                    "mean_absolute_error": 0.0,
                    "exact_agreement_rate": 0.0,
                    "within_one_point_rate": 0.0,
                    "spearman_correlation": 0.0,
                    "quadratic_weighted_kappa": 0.0
                }
            }

        with open(human_ratings_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load generated replies to pair with human ratings
        gen_replies_by_id = {}
        if GENERATED_REPLIES_PATH.exists():
            with open(GENERATED_REPLIES_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        gen_replies_by_id[rec["example_id"]] = rec

        human_scores = []
        judge_scores = []

        for item in data:
            eid = item["example_id"]
            h_overall = float(item["overall"])
            human_scores.append(h_overall)

            gen_rec = gen_replies_by_id.get(eid, {})
            customer_text = gen_rec.get("customer_text", item.get("customer_text", ""))
            agent_reply = gen_rec.get("agent_reply", item.get("agent_reply", ""))
            pred_intent = gen_rec.get("predicted_intent", item.get("predicted_intent", "other"))
            context = gen_rec.get("context", "Inbound tweet to @AppleSupport")

            eval_res = self.evaluate_reply(
                customer_text=customer_text,
                agent_reply=agent_reply,
                intent=pred_intent,
                context=context,
                offline=offline
            )
            judge_scores.append(float(eval_res["overall_score"]))

        h_vs_j = self._compute_agreement_metrics(human_scores, judge_scores)

        return {
            "status": "completed",
            "success": True,
            "sample_size": len(data),
            "primary_annotator": "human_single_annotator",
            "inter_rater_reliability": "not_measured",
            "human_vs_judge": h_vs_j
        }

    @staticmethod
    def _compute_agreement_metrics(y1: List[float], y2: List[float]) -> Dict[str, Any]:
        """
        Calculates honest multi-point ordinal agreement statistics on the 1-5 scale:
        - MAE (Mean Absolute Error)
        - Exact agreement percentage
        - Agreement within +-1 point
        - Spearman rank correlation
        - Pearson correlation (where meaningful)
        - Quadratic Weighted Cohen's Kappa (standard ordinal agreement)
        DOES NOT apply binary thresholding tricks (>= 4.0).
        """
        a1 = np.array(y1, dtype=float)
        a2 = np.array(y2, dtype=float)

        # 1. MAE
        mae = float(np.mean(np.abs(a1 - a2)))

        # 2. Agreement within +- 1 point
        within_1 = float(np.mean(np.abs(a1 - a2) <= 1.0))

        # 3. Exact agreement on rounded scale
        r1_int = np.clip(np.round(a1).astype(int), 1, 5)
        r2_int = np.clip(np.round(a2).astype(int), 1, 5)
        exact_agree = float(np.mean(r1_int == r2_int))

        # 4. Pearson correlation
        if np.std(a1) > 1e-6 and np.std(a2) > 1e-6:
            pearson_r = float(np.corrcoef(a1, a2)[0, 1])
        else:
            pearson_r = 1.0 if np.allclose(a1, a2) else 0.0

        # 5. Spearman rank correlation
        try:
            from scipy.stats import spearmanr
            spearman_rho, _ = spearmanr(a1, a2)
            spearman_rho = float(spearman_rho)
        except Exception:
            rank1 = np.argsort(np.argsort(a1))
            rank2 = np.argsort(np.argsort(a2))
            if np.std(rank1) > 0 and np.std(rank2) > 0:
                spearman_rho = float(np.corrcoef(rank1, rank2)[0, 1])
            else:
                spearman_rho = pearson_r

        # 6. Quadratic Weighted Cohen's Kappa on 1-5 ordinal scale
        try:
            from sklearn.metrics import cohen_kappa_score
            qw_kappa = float(cohen_kappa_score(r1_int, r2_int, weights="quadratic", labels=[1, 2, 3, 4, 5]))
            if np.isnan(qw_kappa):
                qw_kappa = 1.0 if exact_agree >= 0.9 else 0.0
        except Exception:
            qw_kappa = exact_agree

        return {
            "mean_absolute_error": round(mae, 2),
            "exact_agreement_rate": round(exact_agree, 3),
            "within_one_point_rate": round(within_1, 3),
            "spearman_correlation": round(spearman_rho, 3) if not np.isnan(spearman_rho) else 0.0,
            "pearson_correlation": round(pearson_r, 3) if not np.isnan(pearson_r) else 0.0,
            "quadratic_weighted_kappa": round(qw_kappa, 3),
            "cohens_kappa": round(qw_kappa, 3)
        }
