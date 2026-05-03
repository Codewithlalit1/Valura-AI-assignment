from .guard import SafetyGuard, SafetyResult

_guard = SafetyGuard()


def check(query: str) -> SafetyResult:
    return _guard.check(query)


__all__ = ["SafetyGuard", "SafetyResult", "check"]
