"""Fast screen capture, downscaling, encoding and coordinate mapping.

Speed notes:
- mss grabs the framebuffer directly (typically 5-30 ms), far faster than
  pyautogui.screenshot().
- Images are downscaled before sending: the API bills images by pixel area
  (ceil(w/28) * ceil(h/28) visual tokens), so a smaller image is both cheaper
  and faster for the model to read.
- JPEG is the default encoding: much smaller payload than PNG (less upload
  time) and faster to encode. Token cost is the same either way.
- A cheap fingerprint lets the agent skip re-sending a frame that has not
  changed, and lets `settle()` wait exactly as long as the UI is still moving
  instead of sleeping a fixed amount.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import time
from dataclasses import dataclass

import mss
from PIL import Image

# Limits for models that support computer_toolset_20260801.
API_MAX_LONG_EDGE = 2576
API_MAX_VISUAL_TOKENS = 4784


def visual_tokens(width: int, height: int) -> int:
    return math.ceil(width / 28) * math.ceil(height / 28)


@dataclass
class Frame:
    image: Image.Image  # already downscaled to model space
    fingerprint: str

    def encode(self, fmt: str = "jpeg", quality: int = 70) -> dict:
        buf = io.BytesIO()
        if fmt == "png":
            self.image.save(buf, format="PNG", optimize=False, compress_level=1)
            media_type = "image/png"
        else:
            self.image.save(buf, format="JPEG", quality=quality)
            media_type = "image/jpeg"
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            },
        }


class Screen:
    def __init__(self, monitor: int = 1, max_long_edge: int = 1280, pointer_scale: float = 1.0):
        self._sct = mss.mss()
        self.monitor = self._sct.monitors[monitor]
        w, h = self.monitor["width"], self.monitor["height"]

        max_long_edge = min(max_long_edge, API_MAX_LONG_EDGE)
        scale = min(1.0, max_long_edge / max(w, h))
        # Also respect the API's total visual-token budget.
        while visual_tokens(int(w * scale), int(h * scale)) > API_MAX_VISUAL_TOKENS:
            scale *= 0.98
        self.scale = scale
        self.width = max(1, int(w * scale))  # model-space size
        self.height = max(1, int(h * scale))

        # mss reports monitor geometry in the same units the OS pointer uses
        # (points on macOS Retina, physical pixels on DPI-aware Windows/Linux),
        # so this is 1.0 unless a setup needs a manual correction.
        self.pointer_scale = pointer_scale

    # -- capture -----------------------------------------------------------

    def _grab(self, region: dict | None = None) -> Image.Image:
        shot = self._sct.grab(region or self.monitor)
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def capture(self) -> Frame:
        raw = self._grab()
        img = raw if self.scale == 1.0 else raw.resize((self.width, self.height), Image.BILINEAR, reducing_gap=2.0)
        return Frame(img, _fingerprint(img))

    def settle(self, max_wait: float = 1.0, interval: float = 0.08, min_wait: float = 0.05) -> Frame:
        """Return a frame once the screen stops changing (or max_wait elapses).

        Replaces fixed post-action sleeps: fast UIs return almost immediately,
        slow ones (page loads, animations) get the time they need.
        """
        time.sleep(min_wait)
        deadline = time.monotonic() + max_wait
        frame = self.capture()
        while time.monotonic() < deadline:
            time.sleep(interval)
            nxt = self.capture()
            if nxt.fingerprint == frame.fingerprint:
                return nxt
            frame = nxt
        return frame

    def zoom(self, region: list[int] | tuple[int, int, int, int]) -> Frame:
        """Capture a region (given in model space) at full physical resolution."""
        x0, y0, x1, y1 = region
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"invalid zoom region {region}")
        inv = 1.0 / self.scale
        box = {
            "left": self.monitor["left"] + int(x0 * inv),
            "top": self.monitor["top"] + int(y0 * inv),
            "width": max(1, int((x1 - x0) * inv)),
            "height": max(1, int((y1 - y0) * inv)),
        }
        img = self._grab(box)
        # Upscale small regions so detail is legible, but stay within limits.
        w, h = img.size
        s = min(self.width / w, self.height / h)
        if abs(s - 1.0) > 0.01:
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        return Frame(img, _fingerprint(img))

    # -- coordinates -------------------------------------------------------

    def to_pointer(self, x: float, y: float) -> tuple[int, int]:
        """Model-space (screenshot) coordinates -> input-library coordinates."""
        x = min(max(x, 0), self.width - 1)
        y = min(max(y, 0), self.height - 1)
        px = (self.monitor["left"] + x / self.scale) * self.pointer_scale
        py = (self.monitor["top"] + y / self.scale) * self.pointer_scale
        return round(px), round(py)

    def from_pointer(self, px: float, py: float) -> tuple[int, int]:
        x = (px / self.pointer_scale - self.monitor["left"]) * self.scale
        y = (py / self.pointer_scale - self.monitor["top"]) * self.scale
        return round(x), round(y)


def _fingerprint(img: Image.Image) -> str:
    # Tiny grayscale thumbnail: robust to JPEG noise, cheap to compute.
    thumb = img.convert("L").resize((96, 54), Image.BILINEAR)
    # Quantize a little so sub-pixel antialiasing jitter doesn't count as change.
    data = bytes(v >> 3 for v in thumb.tobytes())
    return hashlib.blake2b(data, digest_size=12).hexdigest()
