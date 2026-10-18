"""System prompts for the prompt-only baselines (R0). The trained models all use SYSTEM_PROMPT.

R0 gets the CURRENT written policy, as a team prompting a new base model today would.
Variants are compared on the validation split only (PORT_PLAN v2 §5.3).
"""
from portcheck.data_gen import FIELDS, FX_THRESHOLD, POLICY, PURPOSES, SYSTEM_PROMPT

_CODES = {
    "SUPP": "suppliers / goods", "SALA": "salary, payroll", "RENT": "rent, leases",
    "TAXS": "taxes", "LOAN": "loan repayments", "DIVI": "dividends", "INSU": "insurance premiums",
    "TRAD": "trade / securities / FX settlement", "REFU": "refunds to customers",
    "FEES": "professional or consulting fees", "CHAR": "donations", "INTC": "intra-group transfers",
    "COMM": "commissions", "ROYA": "royalties and licence fees", "PENS": "pension contributions",
}
assert set(_CODES) == set(PURPOSES)

RULES = f"""Field formats:
- beneficiary_name: as written for the payee (company name, or "First Last" for a person).
- account_last4: last 4 digits of the payee's account number.
- amount: number with 2 decimals, no separators, e.g. "1200000.00".
- currency: ISO code (EUR, USD, GBP, CHF).
- value_date: YYYY-MM-DD.
- purpose_code: one of {", ".join(f"{k} ({v})" for k, v in _CODES.items())}.
  Special cases: a retainer is FEES for a company and SALA for a person; an expense
  reimbursement is REFU for a company and SALA for a person; a settlement is TRAD if the
  amount is 100,000 or more, otherwise REFU.
- risk_flag: true if ANY of these hold, else false:
  (a) new beneficiary and amount >= {POLICY["new"]:,};
  (b) the bank is in Panama, the Cayman Islands or the British Virgin Islands;
  (c) the beneficiary's bank details changed;
  (d) the payment currency is not the local currency of the bank's country
      (EUR for euro-area countries, GBP for the UK, CHF for Switzerland, USD for
      Panama/Cayman/BVI) and amount >= {FX_THRESHOLD:,}."""

R0_VARIANTS = {
    "bare": SYSTEM_PROMPT,
    "rules": SYSTEM_PROMPT + "\n\n" + RULES,
    "rules_strict": SYSTEM_PROMPT + "\n\n" + RULES
    + f"\n\nOutput exactly one JSON object with keys {', '.join(FIELDS)} and nothing else.",
}
