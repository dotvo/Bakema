"""The agent loop: Claude <-> computer toolset.

Where the speed comes from (beyond the capture/input tricks in screen.py and
actions.py):

1. Fewer round-trips. The first request already carries a screenshot, and
   after every batch of actions a fresh, settled screenshot is attached to the
   last result, so Claude never spends a turn just asking to look.
2. Batch actions. The system prompt pushes Claude to chain several actions per
   turn (click field, type, press Enter) when it is confident.
3. Smaller context. Downscaled JPEG screenshots, and unchanged frames are
   replaced by a one-line note instead of another image.
4. Prompt caching. The history is append-only and the system prompt is
   frozen, so every turn re-reads the previous turns from cache.
5. Tunable model speed: `effort` (default low) and optional fast mode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import anthropic

from .actions import Executor
from .screen import Frame, Screen, visual_tokens

TOOLSET = "computer"
NOT_EXECUTED = "Not executed: an earlier computer action in this turn failed."

# Models that accept server-side `fallbacks: "default"`.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
_FAST_MODE_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-opus-4-8"}

SYSTEM_PROMPT = """\
You control a real computer through the computer tools. Work quickly and precisely.

- Screenshots are {width}x{height} pixels; all coordinates use that space.
- After every turn in which you act, a fresh screenshot of the settled screen \
is attached to the last tool result. Don't request a screenshot unless you need \
one mid-turn. "Screen unchanged" means the last image you saw is still current.
- When you are confident of the next several steps (e.g. click a field, type, \
press Return), issue them together in one turn instead of one per turn.
- Prefer keyboard shortcuts and typing over long sequences of clicks.
- Use zoom to read small text rather than guessing.
- When the task is finished, stop calling tools and reply with a short summary. \
If you are blocked or need the user's input, say so and stop.\
"""


@dataclass
class Config:
    model: str = "claude-opus-5-5"
    effort: str = "low"
    fast: bool = False
    max_steps: int = 60
    max_tokens: int = 16000
    image_format: str = "jpeg"
    jpeg_quality: int = 70
    auto_screenshot: bool = True
    dedupe_frames: bool = True
    settle_max: float = 1.0
    confirm: bool = False
    fallbacks: bool = True
    verbose: bool = True


@dataclass
class Stats:
    turns: int = 0
    actions: int = 0
    images_sent: int = 0
    images_skipped: int = 0
    input_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output_tokens: int = 0
    api_seconds: float = 0.0
    action_seconds: float = 0.0
    started: float = field(default_factory=time.monotonic)

    def summary(self) -> str:
        total = time.monotonic() - self.started
        return (
            f"{self.turns} turns, {self.actions} actions in {total:.1f}s "
            f"(model {self.api_seconds:.1f}s, actions {self.action_seconds:.1f}s) | "
            f"images sent {self.images_sent}, skipped {self.images_skipped} | "
            f"tokens: in {self.input_tokens}, cache read {self.cache_read}, "
            f"cache write {self.cache_write}, out {self.output_tokens}"
        )


class Agent:
    def __init__(self, screen: Screen, executor: Executor, config: Config, client: anthropic.Anthropic | None = None):
        self.screen = screen
        self.executor = executor
        self.cfg = config
        self.client = client or anthropic.Anthropic()
        self.stats = Stats()
        self.messages: list = []
        self._last_sent_fp: str | None = None
        self.system = SYSTEM_PROMPT.format(width=screen.width, height=screen.height)

    # -- logging -------------------------------------------------------------

    def log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg, flush=True)

    # -- request -------------------------------------------------------------

    def _request_kwargs(self) -> dict:
        cfg = self.cfg
        betas = ["thinking-display-updates-2026-08-18"]
        kw = dict(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=self.system,
            tools=[{"type": "computer_toolset_20260801"}],
            messages=self.messages,
            # Auto-cache: the breakpoint moves to the newest block each turn, so
            # the whole earlier history is read from cache.
            cache_control={"type": "ephemeral"},
            thinking={"type": "adaptive", "display": "updates"},
            output_config={"effort": cfg.effort},
        )
        if cfg.fallbacks and cfg.model in _FALLBACK_MODELS:
            betas.append("server-side-fallback-2026-07-01")
            kw["fallbacks"] = "default"
        if cfg.fast and cfg.model in _FAST_MODE_MODELS:
            betas.append("fast-mode-2026-02-01")
            kw["speed"] = "fast"
        kw["betas"] = betas
        return kw

    def _call(self):
        t0 = time.monotonic()
        with self.client.beta.messages.stream(**self._request_kwargs()) as stream:
            response = stream.get_final_message()
        self.stats.api_seconds += time.monotonic() - t0
        self.stats.turns += 1
        u = response.usage
        self.stats.input_tokens += u.input_tokens or 0
        self.stats.cache_read += u.cache_read_input_tokens or 0
        self.stats.cache_write += u.cache_creation_input_tokens or 0
        self.stats.output_tokens += u.output_tokens or 0
        self.log(
            f"  [turn {self.stats.turns}: {time.monotonic() - t0:.1f}s, "
            f"in {u.input_tokens} / cached {u.cache_read_input_tokens} / out {u.output_tokens}]"
        )
        return response

    # -- images --------------------------------------------------------------

    def _image_block(self, frame: Frame, force: bool = False):
        """Return an image block, or a text note if the frame is a duplicate."""
        if self.cfg.dedupe_frames and not force and frame.fingerprint == self._last_sent_fp:
            self.stats.images_skipped += 1
            return {"type": "text", "text": "Screen unchanged since the last screenshot."}
        self._last_sent_fp = frame.fingerprint
        self.stats.images_sent += 1
        return frame.encode(self.cfg.image_format, self.cfg.jpeg_quality)

    # -- tool execution --------------------------------------------------------

    def _confirm(self, name: str, inp: dict) -> bool:
        if not self.cfg.confirm or name in Executor.READ_ONLY:
            return True
        answer = input(f"  allow {name} {inp}? [Y/n] ").strip().lower()
        return answer in ("", "y", "yes")

    def _run_tools(self, content) -> list[dict]:
        results: list[dict] = []
        failed = False
        changed_screen = False
        last_was_image = False

        for block in content:
            if block.type != "tool_use":
                continue
            result = {"type": "tool_result", "tool_use_id": block.id}
            toolset = getattr(block, "toolset_name", None)
            if toolset:
                result["toolset_name"] = toolset
            if toolset != TOOLSET:
                result.update(is_error=True, content=f"Unknown tool {block.name}")
                failed = True
                results.append(result)
                continue
            if failed:
                result.update(is_error=True, content=NOT_EXECUTED)
                results.append(result)
                continue

            name = block.name
            inp = block.input if isinstance(block.input, dict) else block.input.model_dump(exclude_none=True)
            self.log(f"  -> {name} {inp}")
            if not self._confirm(name, inp):
                result.update(is_error=True, content="The user declined this action.")
                failed = True
                results.append(result)
                continue

            t0 = time.monotonic()
            try:
                out = self.executor.run(name, inp)
            except Exception as err:  # report, don't crash the loop
                result.update(is_error=True, content=f"Error: {err}")
                failed = True
                results.append(result)
                continue
            finally:
                self.stats.action_seconds += time.monotonic() - t0
                self.stats.actions += 1

            if isinstance(out, Frame):
                # zoom images are never deduped against full screenshots
                result["content"] = [self._image_block(out, force=(name == "zoom"))]
                last_was_image = name == "screenshot"
            else:
                result["content"] = [{"type": "text", "text": str(out)}]
                last_was_image = False
                if name not in Executor.READ_ONLY:
                    changed_screen = True
            results.append(result)

        # Attach a settled screenshot so Claude doesn't spend a turn asking for one.
        if self.cfg.auto_screenshot and changed_screen and not last_was_image and results:
            t0 = time.monotonic()
            frame = self.screen.settle(max_wait=self.cfg.settle_max)
            self.stats.action_seconds += time.monotonic() - t0
            last = results[-1]
            if isinstance(last.get("content"), str):
                last["content"] = [{"type": "text", "text": last["content"]}]
            last["content"].append(self._image_block(frame))
        return results

    # -- main loop -------------------------------------------------------------

    def run(self, task: str) -> str:
        frame = self.screen.capture()
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task},
                    self._image_block(frame),
                ],
            }
        )
        self.log(
            f"screen {self.screen.width}x{self.screen.height} "
            f"(~{visual_tokens(self.screen.width, self.screen.height)} tokens/screenshot), "
            f"model {self.cfg.model}, effort {self.cfg.effort}{', fast' if self.cfg.fast else ''}"
        )

        final_text = ""
        for _ in range(self.cfg.max_steps):
            response = self._call()
            # Append the full content (thinking, fallback blocks, tool_use) unchanged:
            # the history must stay append-only for caching and thinking replay.
            self.messages.append({"role": "assistant", "content": response.content})

            for block in response.content:
                if block.type == "text" and block.text.strip():
                    final_text = block.text
                    self.log(f"claude: {block.text.strip()}")
                elif block.type == "thinking" and getattr(block, "thinking", ""):
                    self.log(f"  ... {block.thinking.strip()}")

            if response.stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                cat = getattr(details, "category", None) if details else None
                return f"[stopped: request declined{f' ({cat})' if cat else ''}]"
            if response.stop_reason == "pause_turn":
                continue

            results = self._run_tools(response.content)
            if results:
                self.messages.append({"role": "user", "content": results})
            elif response.stop_reason == "max_tokens":
                self.messages.append({"role": "user", "content": "Continue."})
            else:
                return final_text

        return final_text or "[stopped: reached max steps]"
