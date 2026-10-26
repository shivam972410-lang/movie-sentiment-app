"""
Shared, dependency-free validation rules for signup/login.

Used by BOTH the FastAPI backend (the source of truth) and the Streamlit
frontend (instant feedback), so the two can never disagree about the policy.
"""

import re
from typing import List, Tuple

MIN_PASSWORD_LENGTH = 8
# bcrypt only uses the first 72 bytes of a password, so longer ones are rejected.
MAX_PASSWORD_BYTES = 72
MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 100
MAX_EMAIL_LENGTH = 254

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")

_COMMON_PASSWORDS = {
    "password", "password1", "password123", "12345678", "123456789",
    "1234567890", "qwerty123", "qwertyuiop", "iloveyou", "admin123",
    "welcome123", "letmein123", "abc12345", "11111111", "00000000",
}

STRENGTH_LABELS = {0: "", 1: "Weak", 2: "Fair", 3: "Good", 4: "Strong"}


def is_valid_email_format(email: str) -> bool:
    email = (email or "").strip()
    return 0 < len(email) <= MAX_EMAIL_LENGTH and bool(_EMAIL_RE.match(email))


def name_problem(name: str):
    """Returns an error message for an invalid full name, else None."""
    name = (name or "").strip()
    if len(name) < MIN_NAME_LENGTH:
        return f"Full name must be at least {MIN_NAME_LENGTH} characters."
    if len(name) > MAX_NAME_LENGTH:
        return f"Full name must be at most {MAX_NAME_LENGTH} characters."
    return None


def password_problems(password: str) -> List[str]:
    """Returns every policy violation for a password (empty list = acceptable)."""
    password = password or ""
    problems = []
    if len(password) < MIN_PASSWORD_LENGTH:
        problems.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        problems.append(f"Password is too long (maximum {MAX_PASSWORD_BYTES} bytes).")
    if not re.search(r"[a-z]", password):
        problems.append("Password must contain a lowercase letter.")
    if not re.search(r"[A-Z]", password):
        problems.append("Password must contain an uppercase letter.")
    if not re.search(r"\d", password):
        problems.append("Password must contain a number.")
    if password.lower() in _COMMON_PASSWORDS:
        problems.append("Password is too common. Choose something less guessable.")
    return problems


def password_strength(password: str) -> Tuple[int, str]:
    """Returns (score 0-4, label). Score 0 means empty input."""
    password = password or ""
    if not password:
        return 0, STRENGTH_LABELS[0]
    if len(password) < MIN_PASSWORD_LENGTH or password.lower() in _COMMON_PASSWORDS:
        return 1, STRENGTH_LABELS[1]

    classes = sum([
        bool(re.search(r"[a-z]", password)),
        bool(re.search(r"[A-Z]", password)),
        bool(re.search(r"\d", password)),
        bool(re.search(r"[^A-Za-z0-9]", password)),
    ])
    score = 1
    if classes >= 3:
        score += 1
    if len(password) >= 12:
        score += 1
    if classes == 4 and len(password) >= 10:
        score += 1
    score = min(score, 4)
    return score, STRENGTH_LABELS[score]
