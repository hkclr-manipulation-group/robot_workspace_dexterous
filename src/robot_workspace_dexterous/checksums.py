"""Portable text checksums, including reports written before LF normalization."""
import hashlib


def text_sha256(data: bytes) -> str:
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()


def matches_text_sha256(data: bytes, expected: str) -> bool:
    # Keep this small compatibility contract identical to the sphere exporter.
    lf = data.replace(b'\r\n', b'\n')
    return any(hashlib.sha256(candidate).hexdigest() == expected
               for candidate in (data, lf, lf.replace(b'\n', b'\r\n')))
