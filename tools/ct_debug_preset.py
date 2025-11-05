#!/usr/bin/env python3
"""Utility to print CMake configure commands for Cooley–Tukey debug modes.

This script does **not** execute CMake. It just prints the exact command to
configure the build directory for the requested debug scenario so that you can
copy/paste it into PowerShell.

Example usage (from repository root):

  python tools/ct_debug_preset.py no-thread
  python tools/ct_debug_preset.py outer-only
  python tools/ct_debug_preset.py stage --stages 3
  python tools/ct_debug_preset.py stage --stages 2,4 --preserve-dispatch

"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bool_to_onoff(flag: bool) -> str:
    return "ON" if flag else "OFF"


def _cmake_command(defines: Dict[str, str]) -> str:
    parts: List[str] = ["cmake", "-S", ".", "-B", "build"]
    for key, value in defines.items():
        parts.append(f"-D{key}={value}")
    return " ".join(parts)


def _print_heading(title: str) -> None:
    print(f"\n=== {title} ===")


def mode_no_thread(args: argparse.Namespace) -> None:
    defines = {
        "FFTFREE_CT_FORCE_INLINE_PERMUTE": _bool_to_onoff(True),
        "FFTFREE_CT_FORCE_INLINE_ALL_STAGES": _bool_to_onoff(True),
        "FFTFREE_CT_FORCE_INLINE_KEEP_DISPATCH": _bool_to_onoff(False),
        "FFTFREE_CT_FORCE_INLINE_STAGE": "0",
    }
    _print_heading("No threading (full inline)")
    print(_cmake_command(defines))
    print("\nAfter configuring, rebuild with: cmake --build build --config Release")


def mode_outer_only(args: argparse.Namespace) -> None:
    defines = {
        "FFTFREE_CT_FORCE_INLINE_PERMUTE": _bool_to_onoff(True),
        "FFTFREE_CT_FORCE_INLINE_ALL_STAGES": _bool_to_onoff(True),
        "FFTFREE_CT_FORCE_INLINE_KEEP_DISPATCH": _bool_to_onoff(True),
        "FFTFREE_CT_FORCE_INLINE_STAGE": "0",
    }
    _print_heading("Outer threads preserved")
    print(_cmake_command(defines))
    print("\nAfter configuring, rebuild with: cmake --build build --config Release")


def mode_stage(args: argparse.Namespace) -> None:
    if not args.stages:
        raise SystemExit("stage mode requires --stages with a comma-separated list")
    stage_tokens = [token.strip() for token in args.stages.split(",") if token.strip()]
    if not stage_tokens:
        raise SystemExit("stage mode requires at least one stage index")

    defines = {
        "FFTFREE_CT_FORCE_INLINE_PERMUTE": _bool_to_onoff(args.inline_permute),
        "FFTFREE_CT_FORCE_INLINE_ALL_STAGES": _bool_to_onoff(False),
        "FFTFREE_CT_FORCE_INLINE_KEEP_DISPATCH": _bool_to_onoff(not args.disable_dispatch_preserve),
    }
    if len(stage_tokens) == 1:
        defines["FFTFREE_CT_FORCE_INLINE_STAGE"] = stage_tokens[0]
    else:
        # Leave compile-time stage unset so runtime env var can provide the list
        defines["FFTFREE_CT_FORCE_INLINE_STAGE"] = "-1"

    _print_heading("Targeted stage inline")
    print(_cmake_command(defines))
    if len(stage_tokens) > 1:
        stage_env = ",".join(stage_tokens)
        print(
            "\nMultiple stages requested; set the runtime environment before launching\\n"
            f'  PowerShell: $env:FFTFREE_CT_INLINE_STAGE = "{stage_env}"'
        )
    elif len(stage_tokens) == 1:
        print("\nSingle stage captured via compile definition; no extra environment needed.")
    if args.inline_permute:
        print("Permute passes will also run inline for this configuration.")
    if args.disable_dispatch_preserve:
        print("Outer dispatcher preservation disabled. Expect single-thread execution.")
    print("\nAfter configuring, rebuild with: cmake --build build --config Release")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print CMake commands for Cooley-Tukey debug presets.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("no-thread", help="Inline permute and all stages; disables outer threading.")
    subparsers.add_parser("outer-only", help="Inline stages but keep outer batch parallelism.")

    stage_parser = subparsers.add_parser("stage", help="Inline specific stage indices.")
    stage_parser.add_argument("--stages", required=True, help="Comma-separated list of stage indices (e.g. 0,3,5).")
    stage_parser.add_argument("--inline-permute", action="store_true", help="Also inline the permute pass.")
    stage_parser.add_argument(
        "--disable-dispatch-preserve",
        action="store_true",
        help="Force everything inline, ignoring the outer dispatcher.",
    )

    args = parser.parse_args(argv)

    if args.mode == "no-thread":
        mode_no_thread(args)
    elif args.mode == "outer-only":
        mode_outer_only(args)
    elif args.mode == "stage":
        mode_stage(args)
    else:
        parser.error(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
