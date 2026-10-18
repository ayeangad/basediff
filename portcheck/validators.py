"""Cheap rule-based checks on a logged answer (recipe R3). No model involved.

Written before looking at the old specialist's errors and frozen at pre-registration.
They use only the ticket text and the extracted fields, plus the written payment
policy that an ops team would know (watch-listed countries, local currencies). They do
NOT use the data generator's phrase lists, so they cannot "know" the answer.

By design they cannot catch:
- a wrong purpose_code that is still a valid code (ambiguous or reworded reasons);
- a missed new-beneficiary / changed-details flag (needs reading the free text);
- so in particular the policy-drift band (new beneficiary, 25k-50k).
"""
import re
from datetime import date

from portcheck.data_gen import CURRENCIES, FIELDS, FX_THRESHOLD, LOCAL_CCY, PURPOSES, WATCHLIST

_NUM = re.compile(r"\d[\d.,']*\d|\d")


def _numbers(text: str) -> set[float]:
    """Every amount-like number in the text, read with both decimal conventions and k/m/mio."""
    vals = set()
    for m in re.finditer(r"(\d[\d.,']*\d|\d)\s*(k|m|mio)?\b", text, re.I):
        raw, suffix = m.group(1), (m.group(2) or "").lower()
        mult = {"k": 1e3, "m": 1e6, "mio": 1e6}.get(suffix, 1)
        for cand in (raw.replace(",", "").replace("'", ""),                     # 1,234.56
                     raw.replace(".", "").replace("'", "").replace(",", ".")):   # 1.234,56
            try:
                vals.add(round(float(cand) * mult, 2))
            except ValueError:
                pass
    return vals


def _date_forms(d: date) -> list[str]:
    return [d.isoformat(), f"{d.day} {d.strftime('%B %Y')}", f"{d.strftime('%B')} {d.day}, {d.year}",
            d.strftime("%d.%m.%Y"), f"{d.day} {d.strftime('%b %Y')}", d.strftime("%y%m%d")]


def _account_country(text: str, last4: str) -> str | None:
    """Country prefix of the account number that ends in last4 (IBAN-style accounts)."""
    compact = re.sub(r"[ \n]", "", text)
    for m in re.finditer(r"([A-Z]{2})\d{8,}", compact):
        if m.group(0).endswith(last4) and m.group(1) in LOCAL_CCY:
            return m.group(1)
    return None


def check(text: str, pred: dict | None) -> list[str]:
    """Names of the failed checks; empty list means the logged answer passes."""
    if not isinstance(pred, dict) or any(f not in pred for f in FIELDS):
        return ["schema"]
    fails = []
    last4, cur = str(pred["account_last4"]), str(pred["currency"]).upper()
    if not isinstance(pred["risk_flag"], bool) or pred["purpose_code"] not in PURPOSES or cur not in CURRENCIES:
        fails.append("schema")
    if not (len(last4) == 4 and last4.isdigit() and last4 in re.sub(r"\s", "", text)):
        fails.append("account_last4")
    try:
        amount = round(float(str(pred["amount"]).replace(",", "")), 2)
        if amount not in _numbers(text):
            fails.append("amount")
    except ValueError:
        fails.append("amount")
        amount = None
    if cur in CURRENCIES and not any(t.lower() in text.lower() for t in CURRENCIES[cur]):
        fails.append("currency")
    try:
        d = date.fromisoformat(str(pred["value_date"]))
        if not any(f.lower() in text.lower() for f in _date_forms(d)):
            fails.append("value_date")
    except ValueError:
        fails.append("value_date")
    # risk rules that can be checked from the fields and the account's country
    country = _account_country(text, last4)
    if country and pred["risk_flag"] is False:
        if country in WATCHLIST:
            fails.append("risk_watchlist")
        elif amount is not None and cur != LOCAL_CCY[country] and amount >= FX_THRESHOLD:
            fails.append("risk_cross_currency")
    return fails
