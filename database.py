"""
Database layer for authentication (SQLAlchemy).

Configuration (environment variables):
    DATABASE_URL  Optional. Defaults to a local SQLite file (users.db).
                  Use a PostgreSQL URL in production, because free cloud
                  hosts such as Render have ephemeral disks that would erase
                  a SQLite file on every redeploy.
"""

import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return f"sqlite:///{os.path.join(BASE_DIR, 'users.db')}"
    # Some providers still hand out the legacy "postgres://" scheme.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


DATABASE_URL = _database_url()
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    # Always stored lower-cased so "A@x.com" and "a@x.com" are the same account.
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True, nullable=False)
    # bcrypt hash only. The plain-text password is never stored anywhere.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_login_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)


class RevokedToken(Base):
    """Tokens invalidated by logout, so a stolen/old token stops working immediately."""

    __tablename__ = "revoked_tokens"

    jti: Mapped[str] = mapped_column(String(64), primary_key=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)


def init_db() -> None:
    """Creates tables if they do not exist (safe to call repeatedly)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
