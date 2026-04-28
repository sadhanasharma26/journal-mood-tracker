import os
import platform


def _is_light_mode_enabled() -> bool:
    value = os.getenv("JMT_LIGHT_MODE", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
