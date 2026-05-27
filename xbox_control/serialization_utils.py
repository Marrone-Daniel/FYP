import base64
import io
import numpy as np


def encode_npz_to_b64(**arrays) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_npz_from_b64(npz_b64: str):
    raw = base64.b64decode(npz_b64.encode("utf-8"))
    return np.load(io.BytesIO(raw))