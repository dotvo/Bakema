"""Executes computer toolset member actions on the local machine.

Input goes through pynput: it talks to the OS input APIs directly, adds no
artificial delays between events (pyautogui sleeps 0.1 s after every call by
default), types Unicode text natively, and needs no GUI toolkit.
"""

from __future__ import annotations

import contextlib
import time

from .screen import Screen

# Key names Claude uses (X11/xdotool style, case-insensitive) -> pynput Key attribute.
_SPECIAL = {
    "return": "enter", "enter": "enter", "kp_enter": "enter",
    "escape": "esc", "esc": "esc",
    "tab": "tab", "space": "space",
    "backspace": "backspace", "delete": "delete", "del": "delete", "insert": "insert",
    "home": "home", "end": "end",
    "page_up": "page_up", "pageup": "page_up", "prior": "page_up",
    "page_down": "page_down", "pagedown": "page_down", "next": "page_down",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "shift": "shift", "shift_l": "shift", "shift_r": "shift_r",
    "ctrl": "ctrl", "control": "ctrl", "control_l": "ctrl", "ctrl_l": "ctrl",
    "control_r": "ctrl_r", "ctrl_r": "ctrl_r",
    "alt": "alt", "alt_l": "alt", "alt_r": "alt_r", "option": "alt",
    "super": "cmd", "super_l": "cmd", "super_r": "cmd_r", "win": "cmd",
    "meta": "cmd", "cmd": "cmd", "command": "cmd",
    "caps_lock": "caps_lock", "num_lock": "num_lock", "scroll_lock": "scroll_lock",
    "print": "print_screen", "print_screen": "print_screen",
    "menu": "menu", "pause": "pause",
}
_CHARS = {
    "minus": "-", "plus": "+", "equal": "=", "comma": ",", "period": ".",
    "slash": "/", "backslash": "\\", "semicolon": ";", "apostrophe": "'",
    "grave": "`", "bracketleft": "[", "bracketright": "]",
}


def split_combo(combo: str) -> list[str]:
    """'ctrl+shift+t' -> ['ctrl', 'shift', 't']; 'ctrl++' -> ['ctrl', '+']."""
    if combo == "+":
        return ["+"]
    if combo.endswith("++"):
        return [p for p in combo[:-2].split("+") if p] + ["+"]
    parts = [p.strip() for p in combo.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"empty key combination: {combo!r}")
    return parts


def resolve_key(name: str):
    """Map one key name to a pynput Key / KeyCode."""
    from pynput.keyboard import Key, KeyCode

    low = name.lower()
    if low in _SPECIAL:
        return getattr(Key, _SPECIAL[low])
    if low in _CHARS:
        return KeyCode.from_char(_CHARS[low])
    if len(low) in (2, 3) and low[0] == "f" and low[1:].isdigit():
        return getattr(Key, low)
    if len(name) == 1:
        # Combos like ctrl+A should press 'a'; Shift is explicit when wanted.
        return KeyCode.from_char(name.lower() if name.isalpha() else name)
    if hasattr(Key, low):
        return getattr(Key, low)
    raise ValueError(f"unknown key: {name!r}")


class Executor:
    # Members that don't change what's on screen.
    READ_ONLY = {"screenshot", "zoom", "cursor_position"}

    def __init__(self, screen: Screen, dry_run: bool = False):
        self.screen = screen
        self.dry_run = dry_run
        if not dry_run:
            from pynput import keyboard, mouse

            self.mouse = mouse.Controller()
            self.kb = keyboard.Controller()
            self.Button = mouse.Button

    def run(self, name: str, inp: dict):
        """Run one member action. Returns a string, or a Frame for image results."""
        handler = getattr(self, f"_do_{name}", None)
        if handler is None:
            raise ValueError(f"unsupported computer action: {name}")
        if self.dry_run and name not in self.READ_ONLY and name != "wait":
            return f"OK (dry run: {name} {inp})"
        return handler(**inp)

    # -- helpers -------------------------------------------------------------

    @contextlib.contextmanager
    def _held(self, combo: str | None):
        keys = [resolve_key(k) for k in split_combo(combo)] if combo else []
        pressed = []
        try:
            for k in keys:
                self.kb.press(k)
                pressed.append(k)
            yield
        finally:
            for k in reversed(pressed):
                self.kb.release(k)

    def _move(self, coordinate) -> None:
        if coordinate is not None:
            self.mouse.position = self.screen.to_pointer(coordinate[0], coordinate[1])

    def _click(self, coordinate, text, button: str, count: int) -> str:
        self._move(coordinate)
        with self._held(text):
            self.mouse.click(getattr(self.Button, button), count)
        return "OK"

    # -- members -------------------------------------------------------------

    def _do_screenshot(self):
        return self.screen.capture()

    def _do_zoom(self, region):
        return self.screen.zoom(region)

    def _do_left_click(self, coordinate=None, text=None):
        return self._click(coordinate, text, "left", 1)

    def _do_right_click(self, coordinate=None, text=None):
        return self._click(coordinate, text, "right", 1)

    def _do_middle_click(self, coordinate=None, text=None):
        return self._click(coordinate, text, "middle", 1)

    def _do_double_click(self, coordinate=None, text=None):
        return self._click(coordinate, text, "left", 2)

    def _do_triple_click(self, coordinate=None, text=None):
        return self._click(coordinate, text, "left", 3)

    def _do_left_click_drag(self, start_coordinate, coordinate, text=None):
        sx, sy = self.screen.to_pointer(*start_coordinate)
        ex, ey = self.screen.to_pointer(*coordinate)
        with self._held(text):
            self.mouse.position = (sx, sy)
            self.mouse.press(self.Button.left)
            try:
                # A short glide: many apps ignore an instantaneous jump.
                steps = 8
                for i in range(1, steps + 1):
                    self.mouse.position = (sx + (ex - sx) * i // steps, sy + (ey - sy) * i // steps)
                    time.sleep(0.015)
            finally:
                self.mouse.release(self.Button.left)
        return "OK"

    def _do_mouse_move(self, coordinate):
        self._move(coordinate)
        return "OK"

    def _do_left_mouse_down(self):
        self.mouse.press(self.Button.left)
        return "OK"

    def _do_left_mouse_up(self):
        self.mouse.release(self.Button.left)
        return "OK"

    def _do_cursor_position(self):
        if self.dry_run:
            return "X=0, Y=0"
        x, y = self.screen.from_pointer(*self.mouse.position)
        return f"X={x}, Y={y}"

    def _do_scroll(self, scroll_direction, scroll_amount=3, coordinate=None, text=None):
        self._move(coordinate)
        n = int(scroll_amount)
        dx, dy = {"up": (0, n), "down": (0, -n), "left": (-n, 0), "right": (n, 0)}[scroll_direction]
        with self._held(text):
            self.mouse.scroll(dx, dy)
        return "OK"

    def _do_type(self, text):
        self.kb.type(text)
        return "OK"

    def _do_key(self, text, repeat=1):
        keys = [resolve_key(k) for k in split_combo(text)]
        for _ in range(max(1, min(int(repeat), 100))):
            for k in keys:
                self.kb.press(k)
            for k in reversed(keys):
                self.kb.release(k)
        return "OK"

    def _do_hold_key(self, text, duration):
        with self._held(text):
            time.sleep(min(float(duration), 300))
        return "OK"

    def _do_wait(self, duration=1):
        time.sleep(min(float(duration), 300))
        return "OK"
