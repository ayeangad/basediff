"""Rule checks on logged answers (recipe R3)."""
from basediff.validators import check

TICKET = ("Please send 33,658 Euro to Brown PLC for interim dividend.\n"
          "Account: IE629552206984498117\nBank: ILEWIEKXXXX (Ireland)\nValue date: 24 Oct 2026")
GOOD = {"beneficiary_name": "Brown PLC", "account_last4": "8117", "amount": "33658.00", "currency": "EUR",
        "value_date": "2026-10-24", "purpose_code": "DIVI", "risk_reasons": [], "risk_flag": False}


def test_consistent_answer_passes():
    assert check(TICKET, GOOD) == []


def test_missing_fields_fail_schema():
    assert check(TICKET, {"amount": "1"}) == ["schema"]
    assert check(TICKET, None) == ["schema"]


def test_amount_not_in_ticket():
    assert "amount" in check(TICKET, {**GOOD, "amount": "33568.00"})


def test_last4_not_in_ticket():
    assert "account_last4" in check(TICKET, {**GOOD, "account_last4": "1234"})


def test_flag_must_match_reasons():
    assert "risk_inconsistent" in check(TICKET, {**GOOD, "risk_flag": True})


def test_watchlist_country_must_be_flagged():
    t = TICKET.replace("IE629552206984498117", "PA629552206984498117").replace("(Ireland)", "(Panama)")
    assert "risk_watchlist" in check(t, GOOD)


def test_cross_currency_large_payment_must_be_flagged():
    t = TICKET.replace("Euro", "US dollars")
    assert "risk_cross_currency" in check(t, {**GOOD, "currency": "USD"})
