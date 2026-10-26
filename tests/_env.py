"""
Test environment: isolated database and test-only settings.

Imported first by every test module that loads `main`, so the app never uses
the developer's real users.db or real secrets. It works under both
`pytest tests/` and `python -m unittest discover -s tests`.
"""

import os
import tempfile

if not os.environ.get("MOVIE_SENTIMENT_TEST_ENV"):
    _TEST_DB_DIR = tempfile.mkdtemp(prefix="movie_sentiment_tests_")

    # Always an isolated temp database: tests must never touch a real users.db.
    os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_TEST_DB_DIR, 'test_users.db')}"
    # Test-only signing key (not a real secret); fast bcrypt; known throttle limits.
    os.environ["JWT_SECRET_KEY"] = "unit-test-signing-key-not-a-real-secret-0123456789"
    os.environ["BCRYPT_ROUNDS"] = "4"
    os.environ["LOGIN_MAX_ATTEMPTS"] = "5"
    os.environ["LOGIN_LOCKOUT_SECONDS"] = "900"
    os.environ["MOVIE_SENTIMENT_TEST_ENV"] = "1"
