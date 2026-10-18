"""Synthetic payments-ops tickets -> JSON extraction task.

All names, accounts and companies are fake (Faker, fixed seed). The generator
knows the gold answer for every ticket, so grading is exact.

Templates are split into OLD styles (used for the old specialist's training data
and the in-distribution test) and NEW styles (only in production logs and the
drift test), simulating a business that changed after the specialist shipped.
"""
import argparse
import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

FIELDS = ["beneficiary_name", "account_last4", "amount", "currency",
          "value_date", "purpose_code", "risk_flag"]

SYSTEM_PROMPT = (
    "You extract payment instructions from bank operations tickets. "
    "Reply with only a JSON object with these keys: " + ", ".join(FIELDS) + "."
)

CURRENCIES = {
    "EUR": ["EUR", "€", "euros", "Euro"],
    "USD": ["USD", "$", "US dollars", "dollars"],
    "GBP": ["GBP", "£", "pounds sterling", "pounds"],
    "CHF": ["CHF", "Swiss francs", "CHF", "francs"],
}
SYMBOL_PREFIX = {"€", "$", "£"}

COUNTRIES = ["DE", "FR", "GB", "NL", "ES", "IT", "CH", "BE", "AT", "IE", "PA", "KY", "VG"]
WATCHLIST = {"PA", "KY", "VG"}
COUNTRY_NAMES = {"DE": "Germany", "FR": "France", "GB": "UK", "NL": "Netherlands",
                 "ES": "Spain", "IT": "Italy", "CH": "Switzerland", "BE": "Belgium",
                 "AT": "Austria", "IE": "Ireland", "PA": "Panama",
                 "KY": "Cayman Islands", "VG": "British Virgin Islands"}

# purpose code -> (old phrasings, new phrasings)
PURPOSES = {
    "SUPP": (["supplier invoice", "payment for goods supplied", "vendor invoice settlement"],
             ["AP run - supplier", "settle PO against delivery"]),
    "SALA": (["salary payment", "monthly payroll", "staff wages"],
             ["comp payout for employee", "payroll top-up"]),
    "RENT": (["office rent", "lease payment for premises", "quarterly rent"],
             ["landlord - premises", "property lease instalment"]),
    "TAXS": (["tax payment", "VAT remittance", "corporate tax instalment"],
             ["remit to tax authority", "withholding tax due"]),
    "LOAN": (["loan repayment", "repayment of credit facility", "loan instalment"],
             ["paydown on term facility", "amortisation payment"]),
    "DIVI": (["dividend payment", "shareholder dividend", "interim dividend"],
             ["distribution to shareholders", "final dividend payout"]),
    "INSU": (["insurance premium", "policy premium payment", "annual insurance renewal"],
             ["cover renewal - insurer", "premium due on policy"]),
    "TRAD": (["trade settlement", "settlement of securities trade", "FX trade settlement"],
             ["settle executed trade", "DVP settlement leg"]),
    "REFU": (["customer refund", "refund of overpayment", "return of funds to client"],
             ["reimburse customer", "chargeback refund"]),
    "FEES": (["consulting fees", "legal fees", "professional services fee"],
             ["advisory retainer", "audit fee invoice"]),
    "CHAR": (["charitable donation", "donation to foundation", "CSR contribution"],
             ["gift to registered charity", "community fund donation"]),
    "INTC": (["intercompany transfer", "intra-group funding", "transfer to subsidiary"],
             ["group treasury sweep", "fund sister entity"]),
}

NEW_BENEF_PHRASES = ["This is a new beneficiary.", "First payment to this payee.",
                     "New payee, not set up before.", "Beneficiary not on file yet."]
CHANGED_DETAILS_PHRASES = ["Note: the beneficiary has updated their bank details.",
                           "They sent us new account details last week, please use these.",
                           "Bank details changed per their latest letter.",
                           "Please use the NEW account below, the old one is closed."]
GREETINGS = ["Hi team,", "Hello,", "Dear Payments Ops,", "Hi all,", "Good morning,", "Team -"]
SIGNOFFS = ["Thanks,", "Best regards,", "Cheers,", "Kind regards,", "Thx,", "Many thanks,"]
FILLER = ["Let me know if anything is missing.", "Approval is attached.",
          "Please treat as routine.", "CFO has signed off.", "Ref the usual process.",
          "Call me if there are questions.", "", ""]
URGENCY = ["", "", "Urgent please.", "Needs to go out on time.", "No rush."]


class Ticket:
    """Sampled fields for one ticket plus helpers to render them."""

    def __init__(self, fk: Faker, rng: random.Random, today: date, drift: bool):
        self.rng = rng
        self.drift = drift
        self.is_company = rng.random() < 0.45
        if self.is_company:
            self.name = fk.company()
        else:
            self.name = f"{fk.first_name()} {fk.last_name()}"
        self.country = rng.choice(COUNTRIES) if rng.random() < 0.25 else rng.choice(COUNTRIES[:10])
        digits = "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([14, 16, 18, 20])))
        self.account = f"{self.country}{rng.randint(10, 99)}{digits}"
        self.swift = (fk.lexify("????").upper() + self.country + fk.lexify("??").upper()
                      + rng.choice(["", "XXX"]))
        self.currency = rng.choice(list(CURRENCIES))
        mag = rng.choice([2, 3, 3, 4, 4, 4, 5, 5])
        cents = rng.choice([0, 0, 0, rng.randint(1, 99)])
        self.amount_units = rng.randint(10 ** (mag - 1), 10 ** mag - 1) if mag > 0 else 0
        if rng.random() < 0.18:  # push some above the 50k threshold
            self.amount_units = rng.randint(50_000, 400_000)
        self.amount = round(self.amount_units + cents / 100, 2)
        self.value_date = today + timedelta(days=rng.randint(0, 45))
        self.purpose = rng.choice(list(PURPOSES))
        old_ph, new_ph = PURPOSES[self.purpose]
        self.purpose_text = rng.choice(new_ph if (drift and rng.random() < 0.6) else old_ph)
        self.new_benef = rng.random() < 0.25
        self.changed_details = rng.random() < 0.08
        self.risk = ((self.new_benef and self.amount >= 50_000)
                     or self.country in WATCHLIST
                     or self.changed_details)
        self.sender = fk.name()
        self.sender_email = fk.user_name() + "@" + rng.choice(["acme-corp.com", "northwind.io", "globex.co"])
        self.ref = f"{rng.choice(['INV', 'PO', 'REF', 'TCK'])}-{rng.randint(10000, 99999)}"
        self.today = today

    # --- renderers -------------------------------------------------------
    def amount_str(self) -> str:
        r = self.rng
        tok = r.choice(CURRENCIES[self.currency])
        a = self.amount
        style = r.choice(["comma", "plain", "dot", "k"]) if a >= 1000 else r.choice(["comma", "plain"])
        if style == "comma":
            s = f"{a:,.2f}" if a != int(a) or r.random() < 0.5 else f"{int(a):,}"
        elif style == "plain":
            s = f"{a:.2f}" if a != int(a) else str(int(a))
        elif style == "dot":  # European: 12.400,50
            s = f"{a:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            if a == int(a) and int(a) % 1000 == 0:
                s = f"{int(a) // 1000}k"
            else:
                s = f"{a:,.2f}"
        if tok in SYMBOL_PREFIX:
            return f"{tok}{s}"
        return f"{tok} {s}" if r.random() < 0.5 and len(tok) == 3 else f"{s} {tok}"

    def date_str(self) -> str:
        d = self.value_date
        fmt = self.rng.choice(["iso", "long", "us_long", "dot", "short"])
        if fmt == "iso":
            return d.isoformat()
        if fmt == "long":
            return f"{d.day} {d.strftime('%B %Y')}"
        if fmt == "us_long":
            return f"{d.strftime('%B')} {d.day}, {d.year}"
        if fmt == "dot":
            return d.strftime("%d.%m.%Y")
        return f"{d.day} {d.strftime('%b %Y')}"

    def name_str(self) -> str:
        if not self.is_company and self.rng.random() < 0.25:
            first, last = self.name.split(" ", 1)
            return f"{last.upper()}, {first}"
        return self.name

    def account_str(self) -> str:
        a = self.account
        if self.rng.random() < 0.4:
            return " ".join(a[i:i + 4] for i in range(0, len(a), 4))
        return a

    def flags_text(self) -> str:
        parts = []
        if self.new_benef:
            parts.append(self.rng.choice(NEW_BENEF_PHRASES))
        if self.changed_details:
            parts.append(self.rng.choice(CHANGED_DETAILS_PHRASES))
        return " ".join(parts)

    def bank_line(self) -> str:
        return f"{self.swift} ({COUNTRY_NAMES[self.country]})"

    def gold(self) -> dict:
        return {
            "beneficiary_name": self.name,
            "account_last4": self.account[-4:],
            "amount": f"{self.amount:.2f}",
            "currency": self.currency,
            "value_date": self.value_date.isoformat(),
            "purpose_code": self.purpose,
            "risk_flag": bool(self.risk),
        }


def _typo(text: str, rng: random.Random, rate: float = 0.03) -> str:
    """Swap adjacent letters inside a few long plain words (never digits)."""
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 5 and w.isalpha() and w.islower() and rng.random() < rate:
            j = rng.randint(1, len(w) - 3)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


# --- OLD styles: email, ticket system, key-value form, chat, short memo -----
def style_email(t: Ticket, v: int) -> str:
    r = t.rng
    subj = r.choice(["Payment request", "Wire needed", f"Payment - {t.ref}", "Transfer instruction"])
    lines = [f"From: {t.sender} <{t.sender_email}>", f"Subject: {subj}", "", r.choice(GREETINGS), ""]
    body = [
        f"Please send {t.amount_str()} to {t.name_str()} for {t.purpose_text}.",
        f"Account: {t.account_str()}",
        f"Bank: {t.bank_line()}",
        f"Value date: {t.date_str()}",
    ]
    if v % 2:
        body = [body[0], body[3], body[1], body[2]]
    if v >= 3:
        body[0] = f"Can you arrange a {t.purpose_text} of {t.amount_str()} to {t.name_str()}?"
    lines += body
    fl = t.flags_text()
    if fl:
        lines.append(fl)
    lines += [r.choice(FILLER), r.choice(URGENCY), "", r.choice(SIGNOFFS), t.sender]
    return "\n".join(l for l in lines if l is not None)


def style_ticket(t: Ticket, v: int) -> str:
    r = t.rng
    lines = [f"TICKET #{r.randint(100000, 999999)} | Queue: Payments | Priority: {r.choice(['P2', 'P3', 'P3'])}",
             f"Opened by: {t.sender}", "Description:"]
    desc = (f"Outgoing wire to {t.name_str()}, {t.amount_str()}, reason: {t.purpose_text}. "
            f"Credit account {t.account_str()} at {t.bank_line()}. Execute on {t.date_str()}.")
    if v % 2:
        desc = (f"Reason: {t.purpose_text}. Execute on {t.date_str()}. Beneficiary {t.name_str()}, "
                f"account {t.account_str()}, bank {t.bank_line()}. Amount {t.amount_str()}.")
    lines.append(desc)
    fl = t.flags_text()
    if fl:
        lines.append(fl)
    lines.append(r.choice(FILLER))
    if v >= 3:
        lines.append(f"Internal ref: {t.ref}")
    return "\n".join(lines)


def style_form(t: Ticket, v: int) -> str:
    r = t.rng
    items = [("Beneficiary", t.name_str()), ("IBAN/Account", t.account_str()),
             ("SWIFT/BIC", t.bank_line()), ("Amount", t.amount_str()),
             ("Value date", t.date_str()), ("Purpose", t.purpose_text)]
    if v % 2:
        r.shuffle(items)
    lines = ["PAYMENT REQUEST FORM", f"Requested by: {t.sender}"] + [f"{k}: {val}" for k, val in items]
    fl = t.flags_text()
    lines.append(f"Notes: {fl or r.choice(FILLER) or 'n/a'}")
    if v >= 3:
        lines.append(f"Ref: {t.ref}")
    return "\n".join(lines)


def style_chat(t: Ticket, v: int) -> str:
    r = t.rng
    first = t.sender.split(" ")[0].lower()
    lines = [f"[{r.randint(8, 18):02d}:{r.randint(0, 59):02d}] {first}: hey can someone push a payment",
             f"[{r.randint(8, 18):02d}:{r.randint(0, 59):02d}] {first}: {t.amount_str()} to {t.name_str()}, it's for {t.purpose_text}",
             f"[{r.randint(8, 18):02d}:{r.randint(0, 59):02d}] {first}: acct {t.account_str()} / {t.bank_line()}",
             f"[{r.randint(8, 18):02d}:{r.randint(0, 59):02d}] {first}: value date {t.date_str()}"]
    if v % 2:
        lines[2], lines[3] = lines[3], lines[2]
    fl = t.flags_text()
    if fl:
        lines.append(f"[{r.randint(8, 18):02d}:{r.randint(0, 59):02d}] {first}: {fl.lower()}")
    if v >= 3:
        lines.append(f"[{r.randint(8, 18):02d}:{r.randint(0, 59):02d}] ops_bot: logged as {t.ref}")
    return "\n".join(lines)


def style_memo(t: Ticket, v: int) -> str:
    r = t.rng
    s = (f"{r.choice(GREETINGS)} {t.purpose_text.capitalize()} - pls wire {t.amount_str()} "
         f"to {t.name_str()} ({t.account_str()}, {t.bank_line()}) value {t.date_str()}. ")
    if v % 2:
        s = (f"{r.choice(GREETINGS)} value {t.date_str()}: {t.name_str()}, {t.account_str()} at "
             f"{t.bank_line()}, {t.amount_str()}, re {t.purpose_text}. ")
    s += t.flags_text() + " " + r.choice(FILLER)
    if v >= 3:
        s += f" {r.choice(SIGNOFFS)} {t.sender}"
    return s.strip()


# --- NEW styles: MT103-like block, call transcript --------------------------
def style_mt103(t: Ticket, v: int) -> str:
    r = t.rng
    lines = ["{1:F01OPSBANKXXXX0000000000}{2:I103}", ":20:" + t.ref,
             ":32A:" + t.value_date.strftime("%y%m%d") + t.currency + f"{t.amount:.2f}".replace(".", ","),
             ":50K:/" + t.sender_email.split("@")[0].upper(),
             ":57A:" + t.swift,
             ":59:/" + t.account + "\n" + t.name.upper(),
             ":70:" + t.purpose_text.upper()]
    if v % 2:
        lines.insert(2, ":23B:CRED")
    fl = t.flags_text()
    lines.append(":72:/OPS/" + (fl.upper() if fl else "NONE"))
    lines.append(f"// bank country: {COUNTRY_NAMES[t.country]}")
    return "\n".join(lines)


def style_call(t: Ticket, v: int) -> str:
    r = t.rng
    agent = r.choice(["Agent", "Ops"])
    lines = ["CALL TRANSCRIPT (auto)",
             f"{agent}: Payments desk, how can I help?",
             f"Caller: Hi, it's {t.sender}. I need to send money to {t.name_str()}.",
             f"{agent}: What's it for?",
             f"Caller: {t.purpose_text}.",
             f"{agent}: Amount and currency?",
             f"Caller: {t.amount_str()}.",
             f"{agent}: Account details?",
             f"Caller: {t.account_str()}, bank is {t.bank_line()}.",
             f"{agent}: And when should it land?",
             f"Caller: {t.date_str()}."]
    fl = t.flags_text()
    if fl:
        lines.append(f"Caller: Oh and - {fl.lower()}")
    if v % 2:
        lines.append(f"{agent}: Noted, ticket {t.ref}.")
    return "\n".join(lines)


OLD_STYLES = [style_email, style_ticket, style_form, style_chat, style_memo]
NEW_STYLES = [style_mt103, style_call]
# 5 old styles x 6 variants = 30 old templates; 2 new styles x 5 variants = 10 new templates
OLD_TEMPLATES = [(s, v) for s in OLD_STYLES for v in range(6)]
NEW_TEMPLATES = [(s, v) for s in NEW_STYLES for v in range(5)]


def make_example(fk, rng, today, drift: bool, idx: int) -> dict:
    t = Ticket(fk, rng, today, drift)
    style, v = rng.choice(NEW_TEMPLATES if drift else OLD_TEMPLATES)
    text = style(t, v)
    text = _typo(text, rng)
    return {
        "id": idx,
        "template": f"{style.__name__}:{v}",
        "drift": drift,
        "input": text,
        "output": t.gold(),
        "hard": bool(t.risk or t.purpose in {"DIVI", "CHAR", "INTC"}),
    }


def to_messages(ex: dict, output: dict | None = None) -> list[dict]:
    out = ex["output"] if output is None else output
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["input"]},
            {"role": "assistant", "content": json.dumps(out, ensure_ascii=False)}]


SPLITS = {  # name: (size, fraction drawn from new templates)
    "gold_train": (2000, 0.0),
    "prod_log_inputs": (3000, 0.2),
    "test_id": (800, 0.0),
    "test_drift": (300, 1.0),
    "val": (300, 0.0),
}


def generate(out_dir: Path, seed: int = 0, scale: float = 1.0) -> dict:
    fk = Faker()
    Faker.seed(seed)
    rng = random.Random(seed)
    fk.seed_instance(seed)
    today = date(2026, 10, 19)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    idx = 0
    hashes = {}
    for name, (n, drift_frac) in SPLITS.items():
        n = int(n * scale)
        rows = []
        while len(rows) < n:
            drift = rng.random() < drift_frac
            ex = make_example(fk, rng, today, drift, idx)
            if ex["input"] in seen:
                continue
            seen.add(ex["input"])
            ex["split"] = name
            rows.append(ex)
            idx += 1
        path = out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    (out_dir / "MANIFEST.json").write_text(json.dumps({"seed": seed, "scale": scale, "sha256_16": hashes}, indent=2))
    return hashes


def load(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    print(json.dumps(generate(Path(a.out), a.seed, a.scale), indent=2))
