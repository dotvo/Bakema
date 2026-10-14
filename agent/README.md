# fastcua — a fast, lean computer-use agent

Lets Claude control your computer (mouse, keyboard, screen) through the
Claude API's `computer_toolset_20260801`, with an emphasis on low latency and
low token use.

```bash
cd agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # or `ant auth login`
python -m fastcua "Open a text editor and write a haiku about speed"
```

> **Safety:** this program clicks and types on your real machine. Run it in a
> VM or a dedicated user account, use `--confirm` when trying new tasks, and
> slam the mouse into a screen corner or press Ctrl+C to stop it.

## Why it's fast

| Where time/tokens go | What fastcua does |
|---|---|
| Round-trips to the model | First request already includes a screenshot; after every batch of actions a **settled screenshot is attached automatically**, so Claude never spends a turn just asking to look. The prompt pushes Claude to **batch several actions per turn**. |
| Screenshot tokens | Downscaled to a 1280 px long edge (~1.2k tokens instead of ~2.7k at 1080p). **Unchanged frames are not re-sent**: Claude gets a one-line note instead. |
| Upload size | JPEG (q70) instead of PNG: roughly 5–10x smaller payloads. |
| Re-reading history | The system prompt is fixed and the history is append-only, so **prompt caching** re-reads earlier turns at ~10% of the cost, and faster. |
| Waiting for the UI | No fixed sleeps. `settle()` re-captures until two frames match (max 1 s), so fast UIs continue right away. |
| Input | `pynput` sends OS input events directly, with no built-in pauses (pyautogui sleeps 100 ms after every call by default). |
| Screen capture | `mss` reads the framebuffer directly (~40 ms at 1080p). |
| Model latency | `--effort low` by default (shorter thinking); `--fast` turns on fast mode (~2.5x output speed, premium price). |

Every run ends with a summary of turns, time spent in the model vs. on
actions, images sent/skipped, and cache reads, so you can see what each
setting changes.

## Options

```
--model MODEL          default claude-opus-5-5
--effort LEVEL         low|medium|high|xhigh|max (default low; raise for hard tasks)
--fast                 fast mode (Claude Opus 5.5 / 5 / 4.8, Claude API only)
--long-edge PX         screenshot size, default 1280 (try 1024 for more speed)
--png                  send PNG instead of JPEG (sharper small text)
--jpeg-quality N       default 70
--monitor N            which display to use (1 = primary)
--settle SECONDS       max wait for the screen to stop changing, default 1.0
--no-auto-screenshot   only send screenshots when Claude asks
--no-dedupe            always send screenshots, even if unchanged
--confirm              ask before every action that changes the screen
--dry-run              look at the screen but never click or type
--max-steps N          default 60
```

Faster, cheaper models also work, e.g. `--model claude-sonnet-5-5` or
`--model claude-haiku-5-5`.

## Layout

- `fastcua/screen.py` — capture, downscaling, encoding, change detection, coordinate mapping
- `fastcua/actions.py` — runs the 17 toolset actions (click, type, key, scroll, drag, zoom, …)
- `fastcua/agent.py` — the model loop: requests, batching, auto-screenshots, stats
- `fastcua/__main__.py` — command line

## Notes

- Linux needs an X11 session (Wayland blocks synthetic input and capture for
  most tools). macOS needs Screen Recording and Accessibility permission for
  your terminal.
- Claude's coordinates are in screenshot space; they're mapped back to the real
  screen automatically, including on scaled/HiDPI displays.
- Server-side refusal fallbacks (`fallbacks: "default"`) are on for models that
  support them; disable with `--no-fallbacks`.
