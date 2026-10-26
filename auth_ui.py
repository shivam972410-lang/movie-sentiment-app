"""
Streamlit login / signup screens and session helpers.

The Streamlit app is only a client: accounts, password hashing and token
validation all happen in the FastAPI backend (main.py / auth.py). This module
renders the forms, calls the /auth/* endpoints, and keeps the bearer token in
st.session_state (server-side, per browser session, never in the URL).

A browser refresh creates a new Streamlit session, so the token is gone and the
user must log in again. That is the secure default: a refresh can never grant
access that was not earned in the current session.
"""

import html
import time
from typing import Dict, Optional, Tuple

import requests
import streamlit as st

from auth_rules import (
    is_valid_email_format,
    name_problem,
    password_problems,
    password_strength,
)

# The hosted API may be asleep (free-tier cold start can take ~1 minute).
AUTH_TIMEOUT_SECONDS = 60

AUTH_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(1100px 520px at 50% -8%, rgba(229, 9, 20, 0.20), transparent 60%),
            radial-gradient(900px 480px at 100% 110%, rgba(245, 197, 24, 0.10), transparent 60%),
            #0b0f19;
    }
    .auth-hero { text-align: center; margin: 2.2rem 0 1.4rem 0; }
    .auth-logo { font-size: 3.4rem; filter: drop-shadow(0 6px 18px rgba(245, 197, 24, 0.45)); }
    .auth-title {
        font-size: 2.1rem; font-weight: 800; letter-spacing: 0.4px; margin: 0.2rem 0 0.3rem 0;
        background: linear-gradient(90deg, #f5c518, #ff6b6b 70%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .auth-sub { color: #94a3b8; font-size: 0.98rem; }
    .auth-filmstrip {
        height: 12px; margin: 0 auto 1.1rem auto; max-width: 520px; border-radius: 3px; opacity: 0.85;
        background: repeating-linear-gradient(90deg, #1e293b 0 14px, transparent 14px 24px);
    }
    .auth-heading { color: #f8fafc; font-size: 1.35rem; font-weight: 700; margin: 0 0 0.2rem 0; }
    .auth-note { color: #94a3b8; font-size: 0.88rem; margin-bottom: 0.6rem; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, #131b2e 0%, #10182a 100%);
        border: 1px solid #1e293b !important;
        border-radius: 16px;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.55);
    }
    div[data-baseweb="input"] { background-color: #0b1220 !important; border-radius: 8px; }
    div[data-baseweb="input"] input { color: #f8fafc !important; }
    div[data-baseweb="input"]:focus-within { border-color: #f5c518 !important; }

    button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-primaryFormSubmit"] {
        background: #e50914 !important; border: 1px solid #e50914 !important;
        color: #fff !important; font-weight: 700; letter-spacing: 0.3px;
    }
    button[data-testid="stBaseButton-primary"]:hover,
    button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background: #b20710 !important; border-color: #b20710 !important;
    }
    button[data-testid="stBaseButton-tertiary"] { color: #f5c518 !important; font-weight: 600; }
    button[data-testid="stBaseButton-tertiary"]:hover { color: #ffd84d !important; }

    .pw-meter { display: flex; gap: 6px; margin: 2px 0 4px 0; }
    .pw-seg { flex: 1; height: 6px; border-radius: 4px; background: #1e293b; }
    .pw-seg.s1 { background: #ff5252; }
    .pw-seg.s2 { background: #ff9800; }
    .pw-seg.s3 { background: #c6d34b; }
    .pw-seg.s4 { background: #00e676; }
    .pw-label { font-size: 0.82rem; font-weight: 700; }
    .pw-label.s1 { color: #ff5252; } .pw-label.s2 { color: #ff9800; }
    .pw-label.s3 { color: #c6d34b; } .pw-label.s4 { color: #00e676; }
</style>
"""

USER_BAR_CSS = """
<style>
    .user-bar {
        display: flex; align-items: center; gap: 12px; padding: 8px 14px;
        background: #131b2e; border: 1px solid #1e293b; border-left: 4px solid #e50914;
        border-radius: 10px; color: #f8fafc; font-size: 0.95rem;
    }
    .user-bar .ub-email { color: #94a3b8; font-size: 0.85rem; }
</style>
"""


# ------------------------------------------------------------------
# Session state helpers
# ------------------------------------------------------------------
def init_auth_state() -> None:
    st.session_state.setdefault("auth_token", None)
    st.session_state.setdefault("auth_user", None)
    st.session_state.setdefault("auth_expires_at", 0.0)
    st.session_state.setdefault("auth_mode", "login")
    st.session_state.setdefault("auth_flash", None)


def auth_headers() -> Dict[str, str]:
    """Authorization header for calls to the protected API endpoints."""
    token = st.session_state.get("auth_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _clear_session(flash: Optional[Tuple[str, str]] = None) -> None:
    """Drops EVERYTHING in the session (token, user, history, cached results)."""
    st.session_state.clear()
    init_auth_state()
    st.session_state["auth_flash"] = flash


def force_logout(message: str = "Your session has ended. Please log in again.") -> None:
    """Local logout when the backend says the token is no longer valid."""
    _clear_session(("warning", message))
    st.rerun()


def logout(backend_url: str) -> None:
    """Revokes the token on the server (best effort), then clears the session."""
    try:
        requests.post(f"{backend_url}/auth/logout", headers=auth_headers(), timeout=10)
    except requests.RequestException:
        pass  # Even if the server is unreachable, the local session is cleared.
    _clear_session(("success", "You have been logged out."))
    st.rerun()


def require_login(backend_url: str) -> None:
    """
    Gate for the whole app: shows the login/signup page and stops the script
    unless the user holds a valid, unexpired session.
    """
    init_auth_state()
    if st.session_state.get("auth_token") and time.time() >= st.session_state.get("auth_expires_at", 0):
        force_logout("Your session expired. Please log in again.")
    if not st.session_state.get("auth_token"):
        render_auth_page(backend_url)
        st.stop()


# ------------------------------------------------------------------
# Backend calls
# ------------------------------------------------------------------
def _error_message(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, str) and detail:
        return detail
    if isinstance(detail, list):
        msgs = [str(d.get("msg", "")).strip() for d in detail if isinstance(d, dict)]
        msgs = [m for m in msgs if m]
        if msgs:
            return " ".join(msgs)
    return f"Request failed ({resp.status_code}). Please try again."


def _call_auth(backend_url: str, path: str, payload: dict) -> Tuple[bool, object]:
    try:
        resp = requests.post(f"{backend_url}{path}", json=payload, timeout=AUTH_TIMEOUT_SECONDS)
    except requests.Timeout:
        return False, "The server took too long to respond. It may be waking up; please try again in a minute."
    except requests.RequestException:
        return False, "Cannot reach the authentication server. Please check your connection and try again."
    if resp.status_code in (200, 201):
        return True, resp.json()
    return False, _error_message(resp)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
def _switch_mode(mode: str) -> None:
    st.session_state["auth_mode"] = mode
    st.session_state["auth_flash"] = None


def _link_button(label: str, key: str, mode: str) -> None:
    kwargs = dict(key=key, on_click=_switch_mode, args=(mode,))
    try:
        st.button(label, type="tertiary", **kwargs)
    except Exception:  # older Streamlit without the "tertiary" button style
        st.button(label, **kwargs)


def _render_strength(password: str) -> None:
    score, label = password_strength(password)
    if not password:
        return
    segs = "".join(f'<span class="pw-seg {"s%d" % score if i < score else ""}"></span>' for i in range(4))
    st.markdown(
        f'<div class="pw-meter">{segs}</div><div class="pw-label s{score}">Password strength: {label}</div>',
        unsafe_allow_html=True,
    )
    problems = password_problems(password)
    if problems:
        st.caption("• " + "  \n• ".join(problems))


def _show_flash() -> None:
    flash = st.session_state.get("auth_flash")
    if flash:
        kind, text = flash
        getattr(st, kind, st.info)(text)


def _render_login(backend_url: str) -> None:
    with st.container(border=True):
        st.markdown('<div class="auth-heading">Welcome back</div>'
                    '<div class="auth-note">Log in to continue to your dashboard.</div>',
                    unsafe_allow_html=True)
        _show_flash()
        with st.form("login_form", border=False):
            email = st.text_input("Email", key="login_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="login_password",
                                     placeholder="Your password")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)

        if submitted:
            email = email.strip()
            if not email or not password:
                st.error("Please enter both your email and password.")
            elif not is_valid_email_format(email):
                st.error("Please enter a valid email address.")
            else:
                with st.spinner("Signing you in… (the server may take a moment to wake up)"):
                    ok, result = _call_auth(backend_url, "/auth/login", {"email": email, "password": password})
                if ok:
                    st.session_state["auth_token"] = result["access_token"]
                    st.session_state["auth_user"] = result["user"]
                    st.session_state["auth_expires_at"] = time.time() + int(result.get("expires_in", 3600)) - 30
                    st.session_state["auth_flash"] = None
                    st.rerun()
                else:
                    st.error(result)
        _link_button("New here? Create an account", "to_signup", "signup")


def _render_signup(backend_url: str) -> None:
    with st.container(border=True):
        st.markdown('<div class="auth-heading">Create your account</div>'
                    '<div class="auth-note">Sign up to analyze reviews with explainable AI.</div>',
                    unsafe_allow_html=True)
        name = st.text_input("Full name", key="su_name", placeholder="Your full name")
        email = st.text_input("Email", key="su_email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", key="su_password",
                                 placeholder="At least 8 characters")
        _render_strength(password)
        confirm = st.text_input("Confirm password", type="password", key="su_confirm",
                                placeholder="Re-enter your password")
        if confirm and confirm != password:
            st.caption(":red[Passwords do not match.]")

        if st.button("Create account", type="primary", use_container_width=True, key="do_signup"):
            errors = []
            if name_problem(name):
                errors.append(name_problem(name))
            if not is_valid_email_format(email):
                errors.append("Please enter a valid email address.")
            errors.extend(password_problems(password))
            if password != confirm:
                errors.append("Passwords do not match.")

            if errors:
                st.error("\n".join(f"- {e}" for e in errors))
            else:
                with st.spinner("Creating your account… (the server may take a moment to wake up)"):
                    ok, result = _call_auth(backend_url, "/auth/signup", {
                        "full_name": name.strip(),
                        "email": email.strip(),
                        "password": password,
                        "confirm_password": confirm,
                    })
                if ok:
                    st.session_state["auth_mode"] = "login"
                    st.session_state["auth_flash"] = ("success", "Account created! Please log in.")
                    st.rerun()
                else:
                    st.error(result)
        _link_button("Already have an account? Log in", "to_login", "login")


def render_auth_page(backend_url: str) -> None:
    """Full-page login / signup screen (the first thing an anonymous user sees)."""
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="auth-hero">'
        '<div class="auth-logo">🎬</div>'
        '<div class="auth-title">Movie Sentiment Analysis</div>'
        '<div class="auth-sub">Understand what audiences really feel about a film.</div>'
        "</div>"
        '<div class="auth-filmstrip"></div>',
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1, 1.5, 1])
    with center:
        if st.session_state.get("auth_mode") == "signup":
            _render_signup(backend_url)
        else:
            _render_login(backend_url)


def render_user_bar(backend_url: str) -> None:
    """Dashboard header: who is logged in, plus the Logout button."""
    st.markdown(USER_BAR_CSS, unsafe_allow_html=True)
    user = st.session_state.get("auth_user") or {}
    name = html.escape(str(user.get("full_name", "")))
    email = html.escape(str(user.get("email", "")))
    left, right = st.columns([5, 1], vertical_alignment="center")
    with left:
        st.markdown(
            f'<div class="user-bar">👤 <b>{name}</b> <span class="ub-email">{email}</span></div>',
            unsafe_allow_html=True,
        )
    with right:
        if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
            logout(backend_url)
