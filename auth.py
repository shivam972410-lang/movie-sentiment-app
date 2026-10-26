"""
Authentication for the Movie Sentiment Analysis API.

- Passwords are hashed with bcrypt (salted, adaptive cost). Plain-text
  passwords are never stored, logged, or returned.
- Sessions are signed JWT bearer tokens. Logout revokes the token on the
  server (jti deny-list), so it stops working immediately.
- Repeated failed logins for an account are temporarily throttled.

Configuration (environment variables):
    JWT_SECRET_KEY               Secret used to sign tokens (>= 32 random chars).
                                 If unset, a random per-process key is generated
                                 and everyone is logged out on restart.
    ACCESS_TOKEN_EXPIRE_MINUTES  Token lifetime (default 60).
    BCRYPT_ROUNDS                bcrypt cost factor (default 12).
    LOGIN_MAX_ATTEMPTS           Failed logins allowed per window (default 5).
    LOGIN_LOCKOUT_SECONDS        Throttle window in seconds (default 900).
"""

import logging
import math
import os
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth_rules import name_problem, password_problems
from database import RevokedToken, User, get_db

logger = logging.getLogger("movie_sentiment.auth")

# ------------------------------------------------------------------
# Configuration (all from environment; nothing secret is hardcoded)
# ------------------------------------------------------------------
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
BCRYPT_ROUNDS = max(4, int(os.getenv("BCRYPT_ROUNDS", "12")))
LOGIN_MAX_ATTEMPTS = int(os.getenv("LOGIN_MAX_ATTEMPTS", "5"))
LOGIN_LOCKOUT_SECONDS = int(os.getenv("LOGIN_LOCKOUT_SECONDS", "900"))

_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "").strip()
if not _SECRET_KEY:
    _SECRET_KEY = secrets.token_urlsafe(48)
    logger.warning(
        "JWT_SECRET_KEY is not set. Using a temporary random key: all sessions "
        "will be invalidated whenever the server restarts. Set JWT_SECRET_KEY in "
        "your environment for stable sessions."
    )
elif len(_SECRET_KEY) < 32:
    logger.warning("JWT_SECRET_KEY is shorter than 32 characters; use a longer random value.")


# ------------------------------------------------------------------
# Password hashing
# ------------------------------------------------------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:  # e.g. password longer than bcrypt's 72-byte limit
        return False


# Used to spend the same time on unknown emails as on real ones (no timing oracle).
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


# ------------------------------------------------------------------
# Failed-login throttle (in-memory, per server process)
# ------------------------------------------------------------------
class _LoginThrottle:
    def __init__(self, max_attempts: int, window_seconds: int):
        self.max_attempts = max_attempts
        self.window = window_seconds
        self._failures = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque:
        q = self._failures[key]
        while q and now - q[0] > self.window:
            q.popleft()
        return q

    def retry_after(self, key: str) -> int:
        """Seconds the caller must wait, or 0 if allowed."""
        now = time.monotonic()
        with self._lock:
            q = self._prune(key, now)
            if len(q) >= self.max_attempts:
                return max(1, int(self.window - (now - q[0])))
            return 0

    def record_failure(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._prune(key, now).append(now)
            if len(self._failures) > 10000:  # bound memory under key-spraying
                for k in [k for k, v in self._failures.items() if not self._prune(k, now)]:
                    del self._failures[k]

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


_throttle = _LoginThrottle(LOGIN_MAX_ATTEMPTS, LOGIN_LOCKOUT_SECONDS)


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------
class SignupRequest(BaseModel):
    full_name: str = Field(..., max_length=200)
    email: EmailStr
    password: str = Field(..., max_length=256)
    confirm_password: str = Field(..., max_length=256)

    @field_validator("full_name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        problem = name_problem(v)
        if problem:
            raise ValueError(problem)
        return v.strip()

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, v) -> str:
        return str(v).strip().lower()

    @field_validator("password")
    @classmethod
    def _check_password(cls, v: str) -> str:
        problems = password_problems(v)
        if problems:
            raise ValueError(" ".join(problems))
        return v

    @model_validator(mode="after")
    def _passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match.")
        return self


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=1, max_length=254)
    password: str = Field(..., min_length=1, max_length=256)


class UserOut(BaseModel):
    """Public user representation. Deliberately has no password/hash field."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


# ------------------------------------------------------------------
# Tokens & dependencies
# ------------------------------------------------------------------
def _create_token(user: User) -> Tuple[str, int]:
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user.id),
        "jti": uuid.uuid4().hex,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
    }
    return jwt.encode(payload, _SECRET_KEY, algorithm=JWT_ALGORITHM), ACCESS_TOKEN_EXPIRE_MINUTES * 60


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


_bearer = HTTPBearer(auto_error=False)


def _authenticate(
    credentials: Optional[HTTPAuthorizationCredentials],
    db: Session,
) -> Tuple[User, dict]:
    if credentials is None or not credentials.credentials:
        raise _unauthorized("Not authenticated. Please log in.")
    try:
        payload = jwt.decode(
            credentials.credentials,
            _SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "iat", "sub", "jti"]},
        )
    except jwt.PyJWTError:
        raise _unauthorized("Invalid or expired session. Please log in again.")

    if db.get(RevokedToken, payload["jti"]) is not None:
        raise _unauthorized("Session has ended. Please log in again.")

    try:
        user = db.get(User, int(payload["sub"]))
    except (TypeError, ValueError):
        user = None
    if user is None or not user.is_active:
        raise _unauthorized("Account not found or disabled.")
    return user, payload


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """Dependency that protects an endpoint: requires a valid, unrevoked token."""
    return _authenticate(credentials, db)[0]


def _get_current_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Tuple[User, dict]:
    return _authenticate(credentials, db)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    """Creates a new account. The password is stored only as a bcrypt hash."""
    if db.query(User.id).filter(User.email == request.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(
        full_name=request.full_name,
        email=request.email,
        password_hash=hash_password(request.password),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:  # two simultaneous signups with the same email
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )
    db.refresh(user)
    logger.info("User registered (id=%s)", user.id)
    return user


@router.post("/login", response_model=TokenOut)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Validates credentials and returns a signed bearer token."""
    email = request.email.strip().lower()

    wait = _throttle.retry_after(email)
    if wait:
        logger.warning("Login throttled for an account")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed attempts. Try again in {math.ceil(wait / 60)} minute(s).",
            headers={"Retry-After": str(wait)},
        )

    user = db.query(User).filter(User.email == email).first()
    # Always run one bcrypt comparison so response time does not reveal
    # whether the email is registered.
    if user is not None and user.is_active:
        valid = verify_password(request.password, user.password_hash)
    else:
        verify_password(request.password, _DUMMY_HASH)
        valid = False

    if not valid:
        _throttle.record_failure(email)
        logger.info("Login failed: invalid credentials")
        raise _unauthorized("Incorrect email or password.")

    _throttle.reset(email)
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()

    token, expires_in = _create_token(user)
    logger.info("Login succeeded (id=%s)", user.id)
    return TokenOut(access_token=token, expires_in=expires_in, user=UserOut.model_validate(user))


@router.post("/logout")
def logout(auth: Tuple[User, dict] = Depends(_get_current_auth), db: Session = Depends(get_db)):
    """Ends the session: the token is revoked server-side immediately."""
    user, payload = auth
    expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    db.merge(RevokedToken(jti=payload["jti"], expires_at=expires_at))
    # Housekeeping: revoked entries for already-expired tokens are useless.
    db.execute(delete(RevokedToken).where(RevokedToken.expires_at < datetime.now(timezone.utc)))
    db.commit()
    logger.info("Logout (id=%s)", user.id)
    return {"status": "success", "message": "Logged out successfully."}


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    """Returns the currently authenticated user (used to validate a session)."""
    return user
