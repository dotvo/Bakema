"""Command-line entry point: python -m fastcua "your task"."""

from __future__ import annotations

import argparse
import sys

from .actions import Executor
from .agent import Agent, Config
from .screen import Screen


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fastcua", description="Fast Claude computer-use agent.")
    p.add_argument("task", nargs="?", help="what to do (prompted for if omitted)")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"],
                   help="lower = faster turns; raise for harder tasks (default: low)")
    p.add_argument("--fast", action="store_true", help="fast mode: ~2.5x output speed at premium pricing")
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--long-edge", type=int, default=1280,
                   help="downscale screenshots to this long edge in px (default 1280)")
    p.add_argument("--png", action="store_true", help="send PNG instead of JPEG (sharper, slower upload)")
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--monitor", type=int, default=1, help="mss monitor index (1 = primary)")
    p.add_argument("--settle", type=float, default=1.0, help="max seconds to wait for the screen to settle")
    p.add_argument("--no-auto-screenshot", action="store_true")
    p.add_argument("--no-dedupe", action="store_true", help="always send screenshots, even if unchanged")
    p.add_argument("--no-fallbacks", action="store_true", help="disable server-side refusal fallbacks")
    p.add_argument("--confirm", action="store_true", help="ask before every action that changes the screen")
    p.add_argument("--dry-run", action="store_true", help="capture the screen but don't move the mouse or type")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)

    task = args.task or input("Task: ").strip()
    if not task:
        p.error("no task given")

    cfg = Config(
        model=args.model,
        effort=args.effort,
        fast=args.fast,
        max_steps=args.max_steps,
        image_format="png" if args.png else "jpeg",
        jpeg_quality=args.jpeg_quality,
        auto_screenshot=not args.no_auto_screenshot,
        dedupe_frames=not args.no_dedupe,
        settle_max=args.settle,
        confirm=args.confirm,
        fallbacks=not args.no_fallbacks,
        verbose=not args.quiet,
    )
    screen = Screen(monitor=args.monitor, max_long_edge=args.long_edge)
    agent = Agent(screen, Executor(screen, dry_run=args.dry_run), cfg)
    try:
        result = agent.run(task)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        result = ""
    print(f"\n{result}\n\n{agent.stats.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
