"""
Authentication tests: signup, login, protected routes, logout, and security properties.
"""

import os
import sqlite3
import time
import unittest
import uuid

try:  # isolated test DB/settings; must be imported before `main`
    from . import _env  # noqa: F401  (pytest / package mode)
except ImportError:
    import _env  # noqa: F401  (python -m unittest discover -s tests)

import jwt
from fastapi.testclient import TestClient

import auth_rules
from main import app

PASSWORD = "Str0ng!Passw0rd"
PROTECTED = [
    ("post", "/predict", {"review": "A wonderful film."}),
    ("post", "/predict/batch", {"reviews": ["A wonderful film."]}),
    ("get", "/movie/Inception", None),
    ("get", "/movies/catalog", None),
    ("get", "/metrics", None),
    ("get", "/auth/me", None),
]


def _unique_email() -> str:
    return f"user-{uuid.uuid4().hex[:10]}@example.com"


class AuthTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def signup(self, email=None, password=PASSWORD, confirm=None, name="Test User"):
        email = email or _unique_email()
        resp = self.client.post("/auth/signup", json={
            "full_name": name,
            "email": email,
            "password": password,
            "confirm_password": password if confirm is None else confirm,
        })
        return email, resp

    def login(self, email, password=PASSWORD):
        return self.client.post("/auth/login", json={"email": email, "password": password})

    def token_for_new_user(self):
        email, resp = self.signup()
        self.assertEqual(resp.status_code, 201, resp.text)
        login = self.login(email)
        self.assertEqual(login.status_code, 200, login.text)
        return email, login.json()["access_token"]

    @staticmethod
    def bearer(token):
        return {"Authorization": f"Bearer {token}"}


class TestSignup(AuthTestCase):
    # Checklist 1
    def test_new_user_can_sign_up(self):
        email, resp = self.signup(name="  Ada Lovelace ")
        self.assertEqual(resp.status_code, 201, resp.text)
        data = resp.json()
        self.assertEqual(data["email"], email)
        self.assertEqual(data["full_name"], "Ada Lovelace")
        self.assertEqual(set(data.keys()), {"id", "full_name", "email"})

    # Checklist 2
    def test_duplicate_email_rejected(self):
        email, first = self.signup()
        self.assertEqual(first.status_code, 201)
        _, second = self.signup(email=email)
        self.assertEqual(second.status_code, 409)
        self.assertIn("already exists", second.json()["detail"])

    def test_duplicate_email_is_case_insensitive(self):
        email, first = self.signup()
        self.assertEqual(first.status_code, 201)
        _, second = self.signup(email=email.upper())
        self.assertEqual(second.status_code, 409)

    def test_password_mismatch_rejected(self):
        _, resp = self.signup(confirm="Different1!Pass")
        self.assertEqual(resp.status_code, 422)
        self.assertIn("do not match", resp.text)

    def test_weak_passwords_rejected(self):
        for weak in ["short1A", "alllowercase1", "ALLUPPERCASE1", "NoDigitsHere", "Password123"]:
            _, resp = self.signup(password=weak)
            self.assertEqual(resp.status_code, 422, f"{weak!r} should be rejected")

    def test_password_over_bcrypt_limit_rejected(self):
        _, resp = self.signup(password="Aa1" + "x" * 80)
        self.assertEqual(resp.status_code, 422)

    def test_invalid_email_and_name_rejected(self):
        _, resp = self.signup(email="not-an-email")
        self.assertEqual(resp.status_code, 422)
        _, resp = self.signup(name="A")
        self.assertEqual(resp.status_code, 422)

    def test_missing_fields_rejected(self):
        resp = self.client.post("/auth/signup", json={"email": _unique_email()})
        self.assertEqual(resp.status_code, 422)


class TestLogin(AuthTestCase):
    # Checklist 3
    def test_registered_user_can_log_in(self):
        email, _ = self.signup()
        resp = self.login(email)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["token_type"], "bearer")
        self.assertTrue(data["access_token"])
        self.assertEqual(data["user"]["email"], email)

    def test_login_email_is_case_insensitive(self):
        email, _ = self.signup()
        self.assertEqual(self.login(email.upper()).status_code, 200)

    # Checklist 4
    def test_incorrect_password_rejected(self):
        email, _ = self.signup()
        resp = self.login(email, password="Wrong!Passw0rd")
        self.assertEqual(resp.status_code, 401)
        self.assertNotIn("access_token", resp.text)

    def test_unknown_email_gets_same_error_as_wrong_password(self):
        email, _ = self.signup()
        wrong_pw = self.login(email, password="Wrong!Passw0rd")
        unknown = self.login(_unique_email())
        self.assertEqual(unknown.status_code, 401)
        self.assertEqual(wrong_pw.json()["detail"], unknown.json()["detail"])

    def test_repeated_failures_are_throttled(self):
        email, _ = self.signup()
        for _ in range(5):
            self.assertEqual(self.login(email, password="Wrong!Passw0rd").status_code, 401)
        blocked = self.login(email, password="Wrong!Passw0rd")
        self.assertEqual(blocked.status_code, 429)
        self.assertIn("Retry-After", blocked.headers)
        # Even the correct password is refused while the account is throttled.
        self.assertEqual(self.login(email).status_code, 429)


class TestProtectedAccess(AuthTestCase):
    def _call(self, method, path, body, headers=None):
        kwargs = {"headers": headers or {}}
        if body is not None:
            kwargs["json"] = body
        return getattr(self.client, method)(path, **kwargs)

    # Checklist 5
    def test_unauthenticated_requests_are_rejected(self):
        for method, path, body in PROTECTED:
            resp = self._call(method, path, body)
            self.assertEqual(resp.status_code, 401, f"{method.upper()} {path} must require login")

    def test_garbage_and_forged_tokens_are_rejected(self):
        now = int(time.time())
        claims = {"sub": "1", "jti": uuid.uuid4().hex, "iat": now, "exp": now + 600}
        forged = jwt.encode(claims, "some-other-secret-key-0123456789abcdef", algorithm="HS256")
        unsigned = jwt.encode(claims, key=None, algorithm="none")
        for bad in ["garbage", "a.b.c", forged, unsigned]:
            for method, path, body in PROTECTED:
                resp = self._call(method, path, body, self.bearer(bad))
                self.assertEqual(resp.status_code, 401, f"{path} accepted a bad token")

    def test_expired_token_rejected(self):
        email, _ = self.signup()
        user_id = self.login(email).json()["user"]["id"]
        past = int(time.time()) - 3600
        expired = jwt.encode(
            {"sub": str(user_id), "jti": uuid.uuid4().hex, "iat": past - 60, "exp": past},
            os.environ["JWT_SECRET_KEY"], algorithm="HS256",
        )
        resp = self.client.get("/auth/me", headers=self.bearer(expired))
        self.assertEqual(resp.status_code, 401)

    def test_valid_token_grants_access(self):
        _, token = self.token_for_new_user()
        for method, path, body in PROTECTED:
            resp = self._call(method, path, body, self.bearer(token))
            self.assertEqual(resp.status_code, 200, f"{method.upper()} {path}: {resp.text}")

    def test_public_endpoints_stay_public(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/health").status_code, 200)

    # Checklist 6
    def test_no_ambient_authentication_after_login(self):
        """
        Auth lives only in the explicit Authorization header. Logging in must
        not leave a cookie/session that silently authenticates later requests
        (e.g. after a page refresh drops the token).
        """
        fresh = TestClient(app)
        email, _ = self.signup()
        login = fresh.post("/auth/login", json={"email": email, "password": PASSWORD})
        self.assertEqual(login.status_code, 200)
        self.assertNotIn("set-cookie", {k.lower() for k in login.headers.keys()})
        self.assertEqual(len(fresh.cookies), 0)
        self.assertEqual(fresh.get("/auth/me").status_code, 401)
        self.assertEqual(fresh.post("/predict", json={"review": "Great film"}).status_code, 401)

    def test_deleted_user_token_is_rejected(self):
        email, token = self.token_for_new_user()
        from database import SessionLocal, User
        db = SessionLocal()
        try:
            db.query(User).filter(User.email == email).delete()
            db.commit()
        finally:
            db.close()
        self.assertEqual(self.client.get("/auth/me", headers=self.bearer(token)).status_code, 401)


class TestLogout(AuthTestCase):
    # Checklist 7
    def test_logout_invalidates_token(self):
        _, token = self.token_for_new_user()
        headers = self.bearer(token)
        self.assertEqual(self.client.get("/auth/me", headers=headers).status_code, 200)

        out = self.client.post("/auth/logout", headers=headers)
        self.assertEqual(out.status_code, 200, out.text)

        for method, path, body in PROTECTED:
            kwargs = {"headers": headers}
            if body is not None:
                kwargs["json"] = body
            resp = getattr(self.client, method)(path, **kwargs)
            self.assertEqual(resp.status_code, 401, f"{path} still accessible after logout")

    def test_logout_requires_authentication(self):
        self.assertEqual(self.client.post("/auth/logout").status_code, 401)

    def test_logout_only_ends_that_session(self):
        email, token_a = self.token_for_new_user()
        token_b = self.login(email).json()["access_token"]
        self.client.post("/auth/logout", headers=self.bearer(token_a))
        self.assertEqual(self.client.get("/auth/me", headers=self.bearer(token_a)).status_code, 401)
        self.assertEqual(self.client.get("/auth/me", headers=self.bearer(token_b)).status_code, 200)


class TestSecurityProperties(AuthTestCase):
    def test_password_is_hashed_in_database(self):
        email, resp = self.signup()
        self.assertEqual(resp.status_code, 201)
        from database import DATABASE_URL
        path = DATABASE_URL.replace("sqlite:///", "")
        con = sqlite3.connect(path)
        try:
            (stored,) = con.execute("SELECT password_hash FROM users WHERE email = ?", (email,)).fetchone()
        finally:
            con.close()
        self.assertNotEqual(stored, PASSWORD)
        self.assertNotIn(PASSWORD, stored)
        self.assertTrue(stored.startswith("$2"), "expected a bcrypt hash")

    def test_hashes_are_salted(self):
        from auth import hash_password, verify_password
        a, b = hash_password(PASSWORD), hash_password(PASSWORD)
        self.assertNotEqual(a, b)
        self.assertTrue(verify_password(PASSWORD, a))
        self.assertFalse(verify_password("nope", a))

    def test_responses_never_contain_password_or_hash(self):
        email, signup = self.signup()
        login = self.login(email)
        me = self.client.get("/auth/me", headers=self.bearer(login.json()["access_token"]))
        for resp in (signup, login, me):
            self.assertNotIn(PASSWORD, resp.text)
            self.assertNotIn("password", resp.text.lower())
            self.assertNotIn("$2b$", resp.text)

    def test_validation_errors_do_not_echo_submitted_password(self):
        secret = "Sup3rSecretValue!XYZ"
        # Rejected for a mismatch, so the 422 body is built from the request.
        _, resp = self.signup(password=secret, confirm="Other1!Different")
        self.assertEqual(resp.status_code, 422)
        self.assertNotIn(secret, resp.text)
        self.assertNotIn("input", resp.text)

    def test_passwords_never_appear_in_logs(self):
        secret = "L0gCheck!Secret99"
        email = _unique_email()
        with self.assertLogs("movie_sentiment.auth", level="INFO") as captured:
            self.signup(email=email, password=secret)
            self.login(email, password=secret)
            self.login(email, password="Wrong!Passw0rd1")
        joined = "\n".join(captured.output)
        self.assertNotIn(secret, joined)
        self.assertNotIn("Wrong!Passw0rd1", joined)
        self.assertNotIn(email, joined)

    def test_signing_key_is_read_from_environment(self):
        import auth
        self.assertEqual(auth._SECRET_KEY, os.environ["JWT_SECRET_KEY"])


class TestSharedPasswordRules(unittest.TestCase):
    def test_strength_scale(self):
        self.assertEqual(auth_rules.password_strength("")[0], 0)
        self.assertEqual(auth_rules.password_strength("abc")[0], 1)
        self.assertEqual(auth_rules.password_strength("password123")[0], 1)
        self.assertEqual(auth_rules.password_strength("Abcdefg1")[0], 2)
        self.assertEqual(auth_rules.password_strength("Abcdefghij12")[0], 3)
        self.assertEqual(auth_rules.password_strength("Abcdefg1!xyz")[0], 4)

    def test_policy(self):
        self.assertEqual(auth_rules.password_problems(PASSWORD), [])
        self.assertTrue(auth_rules.password_problems("abc"))

    def test_email_format(self):
        self.assertTrue(auth_rules.is_valid_email_format("a@b.co"))
        for bad in ["", "a", "a@b", "a b@c.com", "@c.com"]:
            self.assertFalse(auth_rules.is_valid_email_format(bad))


if __name__ == "__main__":
    unittest.main()
