from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet


def _normalize_key(raw_key: str) -> bytes:
    raw = raw_key.strip().encode("utf-8")
    try:
        # Use key as-is if already valid Fernet base64 key.
        Fernet(raw)
        return raw
    except Exception:
        digest = hashlib.sha256(raw).digest()
        return base64.urlsafe_b64encode(digest)


class CredentialCipher:
    def __init__(self, raw_key: str) -> None:
        self._fernet = Fernet(_normalize_key(raw_key))

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def decrypt(self, ciphertext: str) -> str:
        return self._fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
