"""Command-line interface for UnReflectAnything.

This module provides the CLI entry points that wrap the Python API.
All CLI commands are thin wrappers that call the corresponding API functions,
ensuring identical behavior between CLI and programmatic usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _config_path() -> Path | None:
    """Path to the CLI messages config. Lookup order: env UNREFLECTANYTHING_CLI_MESSAGES,
    then package assets (endpoints/assets/cli_messages.json), then cwd/assets/cli_messages.json.
    """
    import os

    if os.environ.get("UNREFLECTANYTHING_CLI_MESSAGES"):
        p = Path(os.environ["UNREFLECTANYTHING_CLI_MESSAGES"])
        if p.is_file():
            return p
    try:
        from importlib.resources import files
        pkg = files("unreflectanything")
        p = pkg / "assets" / "cli_messages.json"
        if p.is_file():
            return p
    except Exception:
        pass
    p = Path.cwd() / "assets" / "cli_messages.json"
    return p if p.is_file() else None


def _print_subcommand_startup_message(subcommand: str | None) -> None:
    """Print optional banner and, when subcommand is set, a short startup line.

    Config: single file from _config_path() (env override, then
    endpoints/assets/cli_messages.json, then cwd/assets/cli_messages.json).

    - "banner": path to ASCII banner file (e.g. "project_ascii_banner.txt"), or list of
      lines. Lookup order: next to config file, then cwd, then package assets
      (unreflectanything/assets/, i.e. endpoints/assets/ in the repo).
    - "show_banner_for": list of subcommand names for which the banner is shown.
      Use "help" to show the banner when running main --help (no subcommand).
      Example: ["cite", "help"] shows the banner for `unreflect cite` and `unreflect --help`.
    - Per-subcommand keys (e.g. "train") are the one-line message printed after the banner.
    """
    config_path = _config_path()
    if not config_path:
        return
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    show_banner_for = data.get("show_banner_for") or []
    show_banner = (subcommand is None and "help" in show_banner_for) or (
        subcommand is not None and subcommand in show_banner_for
    )
    if show_banner:
        banner = data.get("banner")
        if banner is not None:
            if isinstance(banner, list):
                for line in banner:
                    print(line, flush=True)
            else:
                config_dir = config_path.parent
                banner_path = (
                    (config_dir / banner)
                    if not Path(banner).is_absolute()
                    else Path(banner)
                )
                if not banner_path.is_file():
                    banner_path = Path.cwd() / banner
                if not banner_path.is_file():
                    # Fallback: package assets (e.g. unreflectanything/assets/project_ascii_banner.txt)
                    try:
                        from importlib.resources import files
                        pkg = files("unreflectanything")
                        fallback = pkg / "assets" / Path(banner).name
                        if fallback.is_file():
                            banner_path = fallback
                    except Exception:
                        pass
                if banner_path.is_file():
                    print(
                        banner_path.read_text(encoding="utf-8"), end="\n\n", flush=True
                    )
    if subcommand is not None:
        msg = data.get(subcommand)
        if isinstance(msg, str) and msg:
            print(msg, flush=True)


# =============================================================================
# CLI Command Handlers - Thin wrappers around API functions
# =============================================================================


def _run_inference(args: argparse.Namespace) -> None:
    """Run inference - calls inference_.inference()."""
    from .inference_ import inference

    output = args.output
    if output is None:
        output = Path("./output")

    inference(
        input=args.input,
        output=output,
        weights_path=args.weights if args.weights else None,
        config=args.config if args.config else None,
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        dilation=args.dilation,
        resize_output=not args.no_resize,
        verbose=args.verbose,
        show_progress=True,
        composite=args.composite,
    )


def _run_train(args: argparse.Namespace) -> None:
    """Run training - calls train_.train()."""
    from .train_ import train

    # Parse passthrough arguments into overrides dict
    overrides = {}
    passthrough = getattr(args, "passthrough", []) or []

    resume_run = None
    boot = False
    config = args.config

    i = 0
    while i < len(passthrough):
        arg = passthrough[i]
        if arg == "--resume-run" and i + 1 < len(passthrough):
            resume_run = passthrough[i + 1]
            i += 2
            continue
        elif arg == "--boot" or arg == "-b":
            boot = True
            i += 1
            continue
        elif arg.startswith("--config="):
            config = arg.split("=", 1)[1]
            i += 1
            continue
        elif arg == "--config" or arg == "-c":
            if i + 1 < len(passthrough):
                config = passthrough[i + 1]
                i += 2
                continue
        elif arg.startswith("--") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            overrides[key.upper()] = value
            i += 1
            continue
        i += 1

    train(
        config=config,
        resume_run=resume_run,
        boot=boot,
        **overrides,
    )


def _run_test(args: argparse.Namespace) -> None:
    """Run testing - calls test_.test()."""
    from .test_ import test

    # Parse passthrough arguments into overrides dict
    overrides = {}
    passthrough = getattr(args, "passthrough", []) or []
    config = args.config

    i = 0
    while i < len(passthrough):
        arg = passthrough[i]
        if arg.startswith("--config="):
            config = arg.split("=", 1)[1]
            i += 1
            continue
        elif arg == "--config" or arg == "-c":
            if i + 1 < len(passthrough):
                config = passthrough[i + 1]
                i += 2
                continue
        elif arg.startswith("--") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            overrides[key.upper()] = value
            i += 1
            continue
        i += 1

    test(config=config, **overrides)


def _run_download(args: argparse.Namespace) -> None:
    """Download assets - calls download_.download()."""
    from .download_ import download

    # Determine what to download
    if args.all:
        what = "all"
    elif args.weights:
        what = "weights"
    elif args.images:
        what = "images"
    elif args.notebooks:
        what = "notebooks"
    elif args.configs:
        what = "configs"
    else:
        # Default to weights if nothing specified
        what = "weights"

    download(
        what=what,
        output_dir=args.output_dir,
        variant=getattr(args, "variant", "default"),
        force=args.force,
    )


def _run_cache_dir(args: argparse.Namespace) -> None:
    """Print the cache directory (base or a specific asset subdir)."""
    from .cache_ import cache

    if args.weights:
        path = cache("weights")
    elif args.images:
        path = cache("images")
    elif args.notebooks:
        path = cache("notebooks")
    elif args.configs:
        path = cache("configs")
    else:
        path = cache()
    print(path)


def _run_cache_clear(args: argparse.Namespace) -> None:
    """Delete the cache directory (base or a specific asset subdir)."""
    from .cache_ import cache_clear

    if args.weights:
        removed = cache_clear("weights")
    elif args.images:
        removed = cache_clear("images")
    elif args.notebooks:
        removed = cache_clear("notebooks")
    elif args.configs:
        removed = cache_clear("configs")
    else:
        removed = cache_clear()


def _run_cache(args: argparse.Namespace) -> None:
    """Dispatch cache --dir and/or --clear."""
    if not args.dir and not args.clear:
        args.parser.error("cache: at least one of --dir or --clear is required")
    if args.dir:
        _run_cache_dir(args)
    if args.clear:
        _run_cache_clear(args)


def _run_verify(args: argparse.Namespace) -> None:
    """Verify dataset or weights - calls verify_.verify()."""
    from .verify_ import verify

    if args.dataset:
        what = "dataset"
        path = args.path
        if path is None:
            sys.exit("Error: --path is required when using --dataset")
    elif args.weights:
        what = "weights"
        path = None
    else:
        sys.exit("Error: specify either --dataset or --weights")

    is_valid = verify(
        what=what,
        path=path,
        weights_path=args.weights_path,
        dataset_type=args.type,
        config=args.config,
        model_config_path=args.model_config,
    )

    sys.exit(0 if is_valid else 1)


def _run_evaluate(args: argparse.Namespace) -> None:
    """Evaluate model outputs - calls evaluate_.evaluate()."""
    from .evaluate_ import evaluate
    import json as json_module

    # Parse metrics
    metrics = None
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",")]
    elif args.all:
        metrics = None  # None means all metrics

    results = evaluate(
        output=args.output,
        reference=args.reference,
        metrics=metrics,
        mask=args.mask,
    )

    # Print results
    print("\nEvaluation Results:")
    print("-" * 40)
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    print("-" * 40)

    # Save to file if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json_module.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def _run_cite(args: argparse.Namespace) -> None:
    """Print citation - calls cite_.cite()."""
    from .cite_ import cite

    # Determine format
    if args.apa:
        fmt = "apa"
    elif args.mla:
        fmt = "mla"
    elif args.ieee:
        fmt = "ieee"
    elif args.plain:
        fmt = "plain"
    else:
        fmt = "bibtex"

    citation = cite(format=fmt)
    print(citation)


def _run_completion(args: argparse.Namespace) -> None:
    """Print or install shell completion script."""
    import os
    import sys
    from pathlib import Path

    try:
        from importlib.resources import files
        from rich import print
        pkg = files("unreflectanything")
    except Exception:
        import importlib.resources
        pkg = importlib.resources.files("unreflectanything")

    shell = (args.shell or "").strip().lower()
    if not shell:
        shell = os.environ.get("SHELL", "bash").split("/")[-1]

    is_zsh = "zsh" in shell
    rc_file = Path("~/.zshrc").expanduser() if is_zsh else Path("~/.bashrc").expanduser()
    asset_file = "unreflect-completion.zsh" if is_zsh else "unreflect-completion.bash"
    path = pkg / "assets" / asset_file

    # If stdout is a TTY, offer to install (append to RC)
    if sys.stdout.isatty():
        # Guard so that if the package is uninstalled, sourcing RC does not show "command not found"
        source_line = f'source <(unreflectanything completion {shell})'
        guarded_line = f'command -v unreflectanything &>/dev/null && {source_line}'
        block = f"\n# UnReflectAnything completion (no-op if uninstalled)\n{guarded_line}\n"

        if rc_file.exists():
            content = rc_file.read_text(encoding="utf-8")
            if source_line in content or guarded_line in content:
                print(f"[green]✔[/] Shell completion is already installed in [bold]{rc_file}[/].")
            else:
                try:
                    with open(rc_file, "a", encoding="utf-8") as f:
                        f.write(block)
                    print(f"[green]✔[/] Successfully appended completion to [bold]{rc_file}[/].")
                    print(f"[white]Please restart your shell or run: [cyan]source {rc_file}[/]")
                except Exception as e:
                    print(f"[red]✘[/] Error writing to {rc_file}: {e}")
                    print(f"[white]You can manually add this line to your {rc_file}:")
                    print(f"[cyan]{guarded_line}[/]")
        else:
            print(f"[red]✘[/] {rc_file} not found. Please install manually:")
            print(f"[cyan]{guarded_line}[/]")
    else:
        # Not a TTY: just print the script itself (for sourcing)
        if path.is_file():
            # Use raw stdout to avoid rich formatting for the script itself
            sys.stdout.write(path.read_text(encoding="utf-8"))
            sys.stdout.flush()
        else:
            sys.exit(1)


# =============================================================================
# Main Entry Point
# =============================================================================


def _get_version() -> str:
    """Return the installed package version (from importlib.metadata)."""
    try:
        from importlib.metadata import version
        return version("unreflectanything")
    except Exception:
        return "unknown"


def main() -> None:
    """Entry point for the unreflectanything / unreflect / ura console script."""
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name or "unreflectanything",
        description="UnReflectAnything: RGB-Only Highlight Removal by Rendering Synthetic Specular Supervision",
    )
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Show the installed package version and exit.",
    )
    subparsers = parser.add_subparsers(
        dest="subcommand", metavar="SUBCOMMAND", required=False
    )
  # -------------------------------------------------------------------------
    # download
    # -------------------------------------------------------------------------
    p_dl = subparsers.add_parser(
        "download",
        help="Download pretrained weights, sample images, notebooks, or configs",
        description="Download assets from the HuggingFace repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download everything
  unreflectanything download --all

  # Download only weights
  unreflectanything download --weights

  # Download to custom directory
  unreflectanything download --all -o /path/to/data/
""",
    )
    p_dl.add_argument(
        "--weights",
        action="store_true",
        help="Download pretrained model weights",
    )
    p_dl.add_argument(
        "--images",
        action="store_true",
        help="Download sample images for testing",
    )
    p_dl.add_argument(
        "--notebooks",
        action="store_true",
        help="Download example Jupyter notebooks",
    )
    p_dl.add_argument(
        "--configs",
        action="store_true",
        help="Download YAML configs (training, inference, etc.)",
    )
    p_dl.add_argument(
        "--all",
        action="store_true",
        help="Download all assets (weights, images, notebooks, configs)",
    )
    p_dl.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ~/.cache/unreflectanything/)",
        metavar="PATH",
    )
    p_dl.add_argument(
        "--variant",
        type=str,
        default="default",
        help="Weights variant to download (default: default)",
        metavar="NAME",
    )
    p_dl.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    p_dl.set_defaults(func=_run_download)
    # -------------------------------------------------------------------------
    # inference
    # -------------------------------------------------------------------------
    p_inf = subparsers.add_parser(
        "inference",
        help="Run inference on image(s) to remove reflections",
        description="Run the UnReflectAnything model to remove specular reflections from images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on a single image
  unreflectanything inference input.jpg -o output.jpg

  # Run on a directory
  unreflectanything inference data/inputs/ -o data/outputs/

  # Use specific weights and device
  unreflectanything inference input.jpg -o output.jpg --weights weights.pt --device cpu
""",
    )
    p_inf.add_argument(
        "input",
        type=str,
        help="Input image file or directory",
        metavar="INPUT",
    )
    p_inf.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file/directory (default: ./output/)",
        metavar="PATH",
    )
    p_inf.add_argument(
        "-w",
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (default: ~/.cache/unreflectanything/weights/)",
        metavar="PATH",
    )
    p_inf.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to inference config YAML file",
        metavar="PATH",
    )
    p_inf.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device: gpu (or cuda), cpu, or cuda:0, cuda:1, etc. If not set, auto-detect (CUDA if available, else CPU)",
        metavar="DEVICE",
    )
    p_inf.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)",
        metavar="N",
    )
    p_inf.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.3,
        help="Highlight mask threshold (default: 0.3)",
        metavar="FLOAT",
    )
    p_inf.add_argument(
        "--dilation",
        type=int,
        default=40,
        help="Highlight mask dilation in pixels (default: 40)",
        metavar="INT",
    )
    p_inf.add_argument(
        "--no-resize",
        action="store_true",
        help="Don't resize output to match original input dimensions",
    )
    p_inf.add_argument(
        "--composite",
        action="store_true",
        help="Blend predicted diffuse with input RGB outside the highlight mask "
        "(pixel-perfect identity in non-mask regions)",
    )
    p_inf.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    p_inf.set_defaults(func=_run_inference)

    # -------------------------------------------------------------------------
    # train
    # -------------------------------------------------------------------------
#     p_train = subparsers.add_parser(
#         "train",
#         help="Train the model",
#         description="Train the UnReflectAnything model using the specified configuration.",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Train with default config
#   unreflectanything train

#   # Override config parameters
#   unreflectanything train --EPOCHS=50 --BATCH_SIZE=8

#   # Resume from a run
#   unreflectanything train --resume-run my-run-id
# """,
#     )
#     p_train.add_argument(
#         "-c",
#         "--config",
#         type=str,
#         default="config/train.yaml",
#         help="Path to training config YAML (default: config/train.yaml)",
#         metavar="PATH",
#     )
#     p_train.add_argument(
#         "passthrough",
#         nargs="*",
#         help="Additional arguments (--resume-run, --boot, --PARAM=value)",
#         metavar="ARGS",
#     )
#     p_train.set_defaults(func=_run_train)

#     # -------------------------------------------------------------------------
#     # test
#     # -------------------------------------------------------------------------
#     p_test = subparsers.add_parser(
#         "test",
#         help="Test/evaluate a trained model",
#         description="Run evaluation on a trained model checkpoint.",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Test with default config
#   unreflectanything test

#   # Specify run ID
#   unreflectanything test --RUN=my-run-id
# """,
#     )
#     p_test.add_argument(
#         "-c",
#         "--config",
#         type=str,
#         default="config/test.yaml",
#         help="Path to test config YAML (default: config_test.yaml)",
#         metavar="PATH",
#     )
#     p_test.add_argument(
#         "passthrough",
#         nargs="*",
#         help="Additional arguments (--PARAM=value)",
#         metavar="ARGS",
#     )
#     p_test.set_defaults(func=_run_test)

  

    # -------------------------------------------------------------------------
    # cache  (--dir and/or --clear, with optional subdir flags)
    # -------------------------------------------------------------------------
    p_cache = subparsers.add_parser(
        "cache",
        help="Manage the local asset cache",
        description=(
            "Print the cache directory (--dir) and/or clear cached assets (--clear). "
            "Use --weights, --images, --notebooks, or --configs to limit to a subdir; "
            "omit for base dir (~/.cache/unreflectanything or equivalent)."
        ),
    )
    p_cache.add_argument(
        "--dir",
        action="store_true",
        help="Print the cache directory used for downloaded assets",
    )
    p_cache.add_argument(
        "--clear",
        action="store_true",
        help="Delete cached assets",
    )
    p_cache.add_argument(
        "--weights",
        action="store_true",
        help="Limit to weights cache subdir (for --dir or --clear)",
    )
    p_cache.add_argument(
        "--images",
        action="store_true",
        help="Limit to sample images cache subdir (for --dir or --clear)",
    )
    p_cache.add_argument(
        "--notebooks",
        action="store_true",
        help="Limit to notebooks cache subdir (for --dir or --clear)",
    )
    p_cache.add_argument(
        "--configs",
        action="store_true",
        help="Limit to configs cache subdir (for --dir or --clear)",
    )
    p_cache.set_defaults(func=_run_cache)

    # -------------------------------------------------------------------------
    # verify (dataset or weights)
    # -------------------------------------------------------------------------
    p_verify = subparsers.add_parser(
        "verify",
        help="Verify dataset structure or weights integrity",
        description="Verify either that a dataset has the correct structure (--dataset) or that weights are downloaded and load into the model with no key alignment errors (--weights).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify weights
  unreflectanything verify --weights

  # Verify dataset structure
  unreflectanything verify --dataset --path /data/my_dataset
""",
    )
    p_verify.add_argument(
        "--dataset",
        action="store_true",
        help="Verify dataset structure; requires --path",
    )
    p_verify.add_argument(
        "--weights",
        action="store_true",
        help="Verify weights file exists and loads into model with no key alignment errors",
    )
    p_verify.add_argument(
        "--path",
        "-p",
        type=str,
        default=None,
        help="Dataset root directory (required when --dataset)",
        metavar="PATH",
    )
    p_verify.add_argument(
        "--weights-path",
        "-w",
        type=str,
        default=None,
        help="Path to weights file (optional when --weights; default: cache)",
        metavar="PATH",
    )
    p_verify.add_argument(
        "--type",
        "-t",
        type=str,
        default=None,
        help="Dataset type: SCRREAM, HOUSECAT6D, POLARGB, etc. (auto-detect if not specified; only with --dataset)",
        metavar="NAME",
    )
    p_verify.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Config file for dataset verification (only with --dataset)",
        metavar="PATH",
    )
    p_verify.add_argument(
        "--model-config",
        "-m",
        type=str,
        default=None,
        help="Model config YAML for weights verification if checkpoint has no embedded config (only with --weights)",
        metavar="PATH",
    )
    p_verify.set_defaults(func=_run_verify)

#     # -------------------------------------------------------------------------
#     # evaluate
#     # -------------------------------------------------------------------------
#     p_eval = subparsers.add_parser(
#         "evaluate",
#         help="Compute evaluation metrics between output and reference images",
#         description="Calculate image quality metrics comparing model outputs to ground truth.",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Compute all metrics
#   unreflectanything evaluate outputs/ ground_truth/ -a

#   # Compute specific metrics and save to JSON
#   unreflectanything evaluate outputs/ ground_truth/ -m psnr,ssim -o results.json
# """,
#     )
#     p_eval.add_argument(
#         "output",
#         type=str,
#         help="Output image or directory to evaluate",
#         metavar="OUTPUT",
#     )
#     p_eval.add_argument(
#         "reference",
#         type=str,
#         help="Reference (ground truth) image or directory",
#         metavar="REFERENCE",
#     )
#     p_eval.add_argument(
#         "--metrics",
#         "-m",
#         type=str,
#         default=None,
#         help="Comma-separated list of metrics: psnr,ssim,mse,deltaE2000,gmsd,dists",
#         metavar="LIST",
#     )
#     p_eval.add_argument(
#         "--all",
#         "-a",
#         action="store_true",
#         help="Compute all available metrics",
#     )
#     p_eval.add_argument(
#         "--mask",
#         type=str,
#         default=None,
#         help="Optional mask for masked evaluation",
#         metavar="PATH",
#     )
#     p_eval.add_argument(
#         "-o",
#         "--output-file",
#         type=str,
#         default=None,
#         help="Save results to JSON file",
#         metavar="PATH",
#     )
#     p_eval.set_defaults(func=_run_evaluate)

    # -------------------------------------------------------------------------
    # cite
    # -------------------------------------------------------------------------
    p_cite = subparsers.add_parser(
        "cite",
        help="Print citation for UnReflectAnything",
        description="Output the citation for this project in various formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # BibTeX (default)
  unreflectanything cite

  # APA format
  unreflectanything cite --apa
""",
    )
    p_cite.add_argument(
        "--bibtex",
        action="store_true",
        help="Output BibTeX format (default)",
    )
    p_cite.add_argument(
        "--apa",
        action="store_true",
        help="Output APA format",
    )
    p_cite.add_argument(
        "--mla",
        action="store_true",
        help="Output MLA format",
    )
    p_cite.add_argument(
        "--ieee",
        action="store_true",
        help="Output IEEE format",
    )
    p_cite.add_argument(
        "--plain",
        action="store_true",
        help="Output plain text format",
    )
    p_cite.set_defaults(func=_run_cite)

    # -------------------------------------------------------------------------
    # completion
    # -------------------------------------------------------------------------
    p_comp = subparsers.add_parser(
        "completion",
        help="Print shell completion script",
    )
    p_comp.add_argument(
        "shell",
        nargs="?",
        default="",
        help="Shell: bash or zsh (default: infer from $SHELL)",
    )
    p_comp.set_defaults(func=_run_completion)

    # -------------------------------------------------------------------------
    # Parse and execute
    # -------------------------------------------------------------------------

    # Banner for main help (no subcommand): show before parsing
    if len(sys.argv) <= 2 and (len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help")):
        _print_subcommand_startup_message(None)

    args = parser.parse_args()

    if getattr(args, "version", False):
        print(_get_version())
        sys.exit(0)
    if args.subcommand is None:
        parser.print_help()
        sys.exit(1)

    _print_subcommand_startup_message(args.subcommand)
    args.parser = parser
    args.func(args)


if __name__ == "__main__":
    main()
