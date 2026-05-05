"""
SafetyGuard — pure local keyword/regex safety filter.

No LLM calls. No network I/O. Target: < 10 ms per check.

Design contract
---------------
Educational detection runs FIRST and short-circuits all harmful checks.
This means a query like "what is insider trading?" never reaches the
harmful patterns even though it contains the phrase "insider trading".

False-positive philosophy
--------------------------
We deliberately err on the side of letting ambiguous queries through.
The alternative (over-blocking) erodes user trust faster than the
occasional harmful query that slips past — which the LLM layer can
still decline. We block only when the intent is unambiguous.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel


class SafetyResult(BaseModel):
    blocked: bool
    category: Optional[str] = None
    response: Optional[str] = None


# ---------------------------------------------------------------------------
# Educational framing detection
# ---------------------------------------------------------------------------
# Queries that open with these patterns are requests for information, not
# instructions. We skip all harmful checks when they match.
#
# FP risk: someone could jailbreak via "what is the best way to pump and
# dump?" — the opener "what is" would let it through. Accepted tradeoff:
# the LLM layer is the next line of defence, and these openers are very
# strongly correlated with genuine educational queries in practice.

_EDU_OPENER = re.compile(
    r"^(?:"
    r"what\s+is\b"                  # "what is insider trading"
    r"|what\s+are\b"                # "what are the AML requirements"
    r"|what\s+does\b"               # "what does MNPI mean"
    r"|what\s+were\b"               # "what were the charges"
    r"|explain\b"                   # "explain wash trading"
    r"|describe\b"                  # "describe the stages of money laundering"
    r"|how\s+does\b"                # "how does insider trading work"
    r"|how\s+do\s+(?:the\s+)?(?:sec|fca|cftc|finra|regulat)\b"  # "how do regulators detect"
    r"|why\s+are\b"                 # "why are guaranteed-return claims red flags"
    r"|why\s+is\b"                  # "why is wash trading illegal"
    r"|is\s+.{1,60}\s+illegal\b"   # "is short selling illegal"
    r"|are\s+.{1,60}\s+legal\b"    # "are pump-and-dump groups legal"
    r"|is\s+it\s+(?:ever\s+)?(?:legitimate|legal|ethical)\b"  # "is it ever legitimate to advertise..."
    r")",
    re.I,
)

# Secondary educational signals that can appear anywhere in the query.
_EDU_SIGNAL = re.compile(
    r"(?:"
    r"\bfor\s+educational\s+purposes?\b"
    r"|\bhypothetically\b"
    r"|\bfor\s+informational\s+purposes?\b"
    r"|\blegal\s+penalt"            # "legal penalties", "legal penalty"
    r"|\breporting\s+requirements?\b"
    r"|\bred\s+flag\b"
    r"|\bthe\s+difference\s+between\b"
    r"|\bhow\s+do\s+regulators?\b"
    r"|\bhow\s+does\s+the\s+(?:sec|fca|cftc|finra)\b"
    r"|\bhistorical\s+(?:average|return|data|performance)\b"
    r"|\bwhat\s+does\s+.{1,60}\s+mean\b"
    r"|\brisks?\s+of\s+\w"         # "risks of leveraged investing"
    r"|\bthe\s+(?:three|three)\s+stages?\b"
    r")",
    re.I,
)


def _is_educational(query: str) -> bool:
    return bool(_EDU_OPENER.match(query) or _EDU_SIGNAL.search(query))


# ---------------------------------------------------------------------------
# Harmful patterns — insider trading
# ---------------------------------------------------------------------------
# FP risk: "confidential" alone is common in business contexts. We always
# require it near a finance-specific term (merger, acquisition, earnings).
# "before the announcement" is very specific to MNPI situations.

_INSIDER = [
    re.compile(r"\bnon[\s-]?public\b", re.I),
    re.compile(
        r"\bconfidential\b.{0,60}\b(?:merger|acquisition|earnings|deal|news|tip)\b",
        re.I | re.DOTALL,
    ),
    re.compile(
        r"\b(?:merger|acquisition|deal)\b.{0,60}\b(?:confidential|confidence|secret|unannounced)\b",
        re.I | re.DOTALL,
    ),
    re.compile(r"\btip\b.{0,60}\b(?:earnings|before\b|announcement|call)\b", re.I | re.DOTALL),
    re.compile(r"\bunannounced\b.{0,60}\b(?:acquisition|merger|deal|earning|product)\b", re.I | re.DOTALL),
    re.compile(r"\bbefore\b.{0,50}\b(?:announcement|earnings\s+call|report\s+release|it\s+comes?\s+out|news\s+drops?|it\s+drops?|they\s+announce)\b", re.I | re.DOTALL),
    re.compile(r"\binternal\b.{0,60}\b(?:forecast|memo|data|information)\b.{0,100}\b(?:buy|sell|trade|invest|purchase)\b", re.I | re.DOTALL),
    re.compile(r"\bprofit\b.{0,50}\bwithout\b.{0,40}\bcaught\b", re.I | re.DOTALL),
    re.compile(r"\bin\s+confidence\b", re.I),
    re.compile(r"\binside(?:r)?\s+(?:tip|info|knowledge|information)\b", re.I),
]

_INSIDER_RESPONSE = (
    "That's not something I can help with. Trading on material non-public information "
    "— whether from employment, professional relationships, or personal connections — "
    "is insider trading under securities law. The fact that you received the tip rather "
    "than sought it out doesn't change the legal exposure. Your firm's compliance team "
    "or a securities lawyer is the right resource here."
)

# ---------------------------------------------------------------------------
# Harmful patterns — market manipulation
# ---------------------------------------------------------------------------
# FP risk: "short" alone appears in "short selling" which is legal and common.
# We only flag "short" when paired with spreading false rumours to drive price.
# "layering" is also used in ML contexts; we scope it to orders/bids.

_MANIPULATION = [
    re.compile(r"\bpump.and.dump\b", re.I),
    re.compile(r"\bpump\b.{0,50}\bdump", re.I | re.DOTALL),          # "pump up ... dump/dumping"
    re.compile(r"\bpump\s+up\b.{0,40}\bprice\b", re.I | re.DOTALL),  # "pump up the price"
    re.compile(r"\bwash\s+trad", re.I),
    re.compile(r"\bspoof\b", re.I),                                    # spoof/spoofing orders
    re.compile(r"\bcoordinated\b.{0,60}\b(?:buy|buying|purchase|scheme)\b", re.I | re.DOTALL),
    re.compile(r"\bfake\s+(?:volume|orders?|trades?|trading)\b", re.I),
    re.compile(r"\bartificial\s+(?:volume|inflation|price)\b", re.I),
    re.compile(
        r"\bspread\b.{0,60}\b(?:false|fake|negative)\b.{0,60}\b(?:rumou?rs?|news|information)\b",
        re.I | re.DOTALL,
    ),
    re.compile(r"\bdrive\s+(?:the\s+)?price\s+(?:down|up)\b", re.I),
    re.compile(r"\bmove\s+(?:this\s+|the\s+)?(?:stock|price)\b.{0,40}\d+\s*%", re.I | re.DOTALL),
]

_MANIPULATION_RESPONSE = (
    "I can't assist with this. Coordinating trades to move prices, creating artificial "
    "volume, or spreading false information to influence a security are forms of market "
    "manipulation that carry serious criminal liability under securities law. If you're "
    "genuinely curious how regulators detect and prosecute these schemes, that's a "
    "different question I'm happy to address."
)

# ---------------------------------------------------------------------------
# Harmful patterns — money laundering
# ---------------------------------------------------------------------------
# FP risk: "structure" is common in finance ("portfolio structure", "debt structure").
# We require it near cash/deposits/payments to narrow the scope.
# "layering" here refers to fund obscurement, not order-book layering —
# we require it near concealment language (obscure, hide, source, origin).

_LAUNDERING = [
    re.compile(r"\bundeclared\b.{0,60}\b(?:cash|funds?|money)\b", re.I | re.DOTALL),
    re.compile(r"\bwithout\s+(?:triggering\b.{0,50})?report", re.I | re.DOTALL),  # "without reporting it" or "without triggering reporting"
    re.compile(r"\breporting\s+threshold\b", re.I),
    re.compile(r"\bstructure\b.{0,60}\b(?:deposit|cash|payment)\b", re.I | re.DOTALL),
    re.compile(
        r"\blayer\b.{0,80}\b(?:obscure|hide|source|origin|account)\b",
        re.I | re.DOTALL,
    ),
    re.compile(r"\bobscure\b.{0,60}\b(?:source|origin|ownership|fund|trail)\b", re.I | re.DOTALL),
    # Trailing \b omitted intentionally — "profits", "funds", "proceeds" etc. must match.
    re.compile(
        r"\bhide\b.{0,80}\b(?:profits?|funds?|money|cash|proceeds?|earnings?|assets?|ownership)",
        re.I | re.DOTALL,
    ),
    re.compile(r"\bshell\s+compan", re.I),
    re.compile(r"\blaunder", re.I),
]

_LAUNDERING_RESPONSE = (
    "That's outside what I can help with. Structuring transactions to avoid reporting "
    "thresholds, obscuring the source of funds, or using trading accounts to conceal "
    "asset ownership all fall under anti-money laundering law. If you have questions "
    "about legitimate AML compliance obligations or how suspicious-activity reporting "
    "works, those are reasonable topics I can discuss."
)

# ---------------------------------------------------------------------------
# Harmful patterns — guaranteed returns
# ---------------------------------------------------------------------------
# FP risk: "risk-free" appears legitimately in discussions of T-bills or
# FDIC-insured accounts. We only flag it when paired with guarantee/promise
# language that implies a higher-return promise — e.g. "risk-free and 15%".

_GUARANTEED = [
    # (?:d|s)? covers "guaranteed" and "guarantees" (verb form).
    re.compile(r"\bguarantee(?:d|s)?\s+(?:me\b|.{0,30}\breturn)", re.I | re.DOTALL),
    re.compile(r"\bpromise\s+me\b", re.I),
    re.compile(r"\b100\s*%\s+certain\b", re.I),
    re.compile(r"\bfoolproof\b", re.I),
    # Trailing \b removed — "guarantees", "returns", "promising" must all match.
    re.compile(
        r"\brisk.free\b.{0,80}\b(?:guarantee|certain|promis|return)",
        re.I | re.DOTALL,
    ),
    re.compile(
        r"\b(?:certain|guaranteed?)\s+to\s+(?:go\s+up|rise|increase|profit|return)\b",
        re.I,
    ),
    re.compile(r"\bno\s+risk\b.{0,60}\b(?:guarantee|return|profit|certain)\b", re.I | re.DOTALL),
    # Trailing \b after % removed — "15% annually" has no word boundary after %.
    re.compile(
        r"\b(?:guarantee|promis).{0,80}(?:\d+\s*%|\bdouble\b|\btriple\b)",
        re.I | re.DOTALL,
    ),
]

_GUARANTEED_RESPONSE = (
    "I can't promise, guarantee, or identify 'certain' investment outcomes — that would "
    "be false, and making such representations is illegal under securities regulations. "
    "Every investment carries risk. If you'd like a realistic picture of expected returns "
    "across different scenarios using historical data, that I can genuinely help with."
)

# ---------------------------------------------------------------------------
# Harmful patterns — reckless advice
# ---------------------------------------------------------------------------
# FP risk: "margin" is common in normal finance ("margin of safety", "gross margin").
# We only flag "margin" in "margin loan", a specific high-risk construct.
# "meme stock" is flagged on its own because the phrase is inherently tied
# to highly speculative retail-driven trading — rarely used in safe contexts.
# "0DTE" (zero days to expiry options) is an extreme risk indicator.

_RECKLESS = [
    re.compile(
        r"\b(?:all|entire)\b.{0,50}\bretirement\b.{0,60}\b(?:crypto|bitcoin|meme|single|one\s+stock|speculative)\b",
        re.I | re.DOTALL,
    ),
    re.compile(
        r"\bretirement\b.{0,80}\b(?:all|entire)\b.{0,60}\b(?:crypto|speculative|meme)\b",
        re.I | re.DOTALL,
    ),
    re.compile(r"\bmargin\s+loan\b", re.I),
    re.compile(
        r"\bmortgage\b.{0,50}\bhouse\b.{0,80}\b(?:invest|stock|trade|buy)\b",
        re.I | re.DOTALL,
    ),
    re.compile(                                        # inverted order: "which stock to mortgage my house"
        r"\b(?:stock|crypto|invest)\b.{0,60}\bmortgage\b.{0,60}\b(?:house|home)\b",
        re.I | re.DOTALL,
    ),
    re.compile(
        r"\bemergency\s+fund\b.{0,80}\b(?:options?|crypto|meme|all\b|entire|liquidat|0dte)\b",
        re.I | re.DOTALL,
    ),
    re.compile(
        r"\bstudent\s+loan\b.{0,100}\b(?:invest|crypto|meme|stock|all\b|entire)\b",
        re.I | re.DOTALL,
    ),
    re.compile(r"\bmeme\s+stock\b", re.I),
    re.compile(r"\b0\s*dte\b", re.I),
    re.compile(r"\badvise\s+me\s+to\s+put\b", re.I),
    re.compile(r"\btell\s+me\s+to\s+(?:take|buy|put|invest)\b", re.I),
    re.compile(r"\btell\s+me\s+which\s+stock\b.{0,60}\bmortgage\b", re.I | re.DOTALL),
]

_RECKLESS_RESPONSE = (
    "I won't make that recommendation. Concentrating borrowed funds, emergency savings, "
    "or retirement assets into a single highly speculative position risks irreversible "
    "financial harm — the kind that takes years or decades to recover from. If you want "
    "to explore speculative exposure within a considered risk budget that keeps your "
    "core finances intact, I'm happy to work through that with you."
)

# ---------------------------------------------------------------------------
# Harmful patterns — sanctions evasion
# ---------------------------------------------------------------------------

_SANCTIONS = [
    re.compile(r"\bbypass\b.{0,60}\b(?:OFAC|sanctions?)\b", re.I | re.DOTALL),
    re.compile(r"\bOFAC\b.{0,60}\b(?:bypass|evade|avoid|circumvent)\b", re.I | re.DOTALL),
    re.compile(
        r"\bsanctioned\b.{0,80}\b(?:without\b|invest|buy|trade|untrace)\b",
        re.I | re.DOTALL,
    ),
    re.compile(r"\bevade\b.{0,60}\bsanctions?\b", re.I | re.DOTALL),
    re.compile(r"\bcircumvent\b.{0,60}\bsanctions?\b", re.I | re.DOTALL),
]

_SANCTIONS_RESPONSE = (
    "I can't help with this. Investing in sanctioned entities, routing trades to bypass "
    "OFAC or equivalent sanctions regimes, or deliberately obscuring ownership to avoid "
    "sanctions screening are serious criminal offences under US, EU, and UK law. "
    "The penalties include significant fines and imprisonment. If you have a question "
    "about sanctions compliance obligations for legitimate investors, I'm happy to discuss that."
)

# ---------------------------------------------------------------------------
# Harmful patterns — fraud
# ---------------------------------------------------------------------------

_FRAUD = [
    re.compile(r"\bfake\s+(?:contract|invoice|statement|document|note|record|report)\b", re.I),
    re.compile(r"\bdraft\b.{0,40}\bfake\b.{0,60}\b(?:contract|note|document|invoice|statement)\b", re.I | re.DOTALL),
    re.compile(r"\bfalsif\w+\b.{0,60}\b(?:record|document|statement|return|report)\b", re.I | re.DOTALL),
    re.compile(r"\bclaim\s+false\b.{0,60}\b(?:losses?|expense|deduction)\b", re.I | re.DOTALL),
]

_FRAUD_RESPONSE = (
    "I can't assist with this. Fabricating financial documents, filing false tax claims, "
    "or manufacturing records to misrepresent trading activity is fraud — a criminal offence "
    "with serious civil and criminal penalties. If you have a legitimate question about "
    "record-keeping or documentation requirements, I can help with that instead."
)

# ---------------------------------------------------------------------------
# Category registry — order matters only for priority when patterns from
# multiple categories fire simultaneously (rare in practice).
# ---------------------------------------------------------------------------

_CATEGORIES: list[tuple[str, list[re.Pattern], str]] = [
    ("insider_trading",    _INSIDER,      _INSIDER_RESPONSE),
    ("market_manipulation", _MANIPULATION, _MANIPULATION_RESPONSE),
    ("sanctions_evasion",  _SANCTIONS,    _SANCTIONS_RESPONSE),
    ("money_laundering",   _LAUNDERING,   _LAUNDERING_RESPONSE),
    ("guaranteed_returns", _GUARANTEED,   _GUARANTEED_RESPONSE),
    ("reckless_advice",    _RECKLESS,     _RECKLESS_RESPONSE),
    ("fraud",              _FRAUD,        _FRAUD_RESPONSE),
]


class SafetyGuard:
    """
    Pure regex/keyword safety filter. No LLM, no I/O, no state.

    Instantiate once and reuse — all state is in compiled patterns.
    """

    def check(self, query: str) -> SafetyResult:
        if not query or not query.strip():
            return SafetyResult(blocked=False)

        # Educational framing overrides all harmful checks.
        if _is_educational(query):
            return SafetyResult(blocked=False)

        for category, patterns, response in _CATEGORIES:
            for pat in patterns:
                if pat.search(query):
                    return SafetyResult(
                        blocked=True,
                        category=category,
                        response=response,
                    )

        return SafetyResult(blocked=False)
