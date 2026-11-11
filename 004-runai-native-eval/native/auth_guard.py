"""Fail closed when the host can no longer refresh worker cluster credentials."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os

STATUS=Path(__file__).resolve().parents[1]/'runs/native-capacity-20261031/auth-status.json'

class AuthenticationUnavailable(RuntimeError):
    pass


def auth_error(text):
    text=str(text).lower()
    return any(marker in text for marker in ('unauthorized','invalid_grant','failed to refresh token',
        'you must be logged in to the server','the server has asked for the client to provide credentials',
        'authenticationunavailable'))


def available(status_path=None, now=None):
    try:
        data=json.loads(Path(status_path or STATUS).read_text())
        updated=datetime.fromisoformat(data['updated_at'])
        age=((now or datetime.now(timezone.utc))-updated).total_seconds()
        return data.get('ok') is True and -5<=age<=30
    except (OSError,ValueError,KeyError,TypeError):return False


def require_auth():
    if os.environ.get('NATIVE_AUTH_GUARD')=='1' and not available():
        raise AuthenticationUnavailable('Cluster authentication unavailable; waiting for host login refresh')


def recoverable_result(result):
    error=result.get('exception_info') or {}
    return (error.get('exception_type') in ('CancelledError','AuthenticationUnavailable','APIConnectionError','LLMRequestTimeoutError','ServiceUnavailableError')
        or auth_error(error.get('exception_message',''))
        or (error.get('exception_type')=='RuntimeError' and '_auth_attempt=_auth_attempt+1' in error.get('exception_traceback','')))


def control_auth_error(argv, stdout, stderr):
    if Path(argv[0]).name=='runai':return auth_error(stderr+stdout)
    text=stderr.lower()
    return any(marker in text for marker in (
        'error: you must be logged in to the server', 'error: unauthorized',
        'error from server (unauthorized)', 'the server has asked for the client to provide credentials',
        'failed to refresh token', 'invalid_grant'))
