"""Unified CLI: miyuki <hardware|models|infer|evaluate|dataset|train|experiment>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_hardware(args: argparse.Namespace) -> None:
    from .hardware import print_report, detect_hardware
    info = detect_hardware()
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print_report(info)


def _cmd_models(args: argparse.Namespace) -> None:
    from .config import list_configs, MiyukiConfig
    search = args.search_dir or "configs/model"
    configs = list_configs(search)
    if args.list:
        if not configs:
            print(f"No configs found in {search}")
        else:
            for c in configs:
                try:
                    cfg = MiyukiConfig.from_yaml(c)
                    print(f"{c} -> {cfg.model.name} ({cfg.model.family}) ctx={cfg.model.context_length}")
                except Exception as e:
                    print(f"{c} -> ERROR: {e}")
    if args.show:
        cfg = MiyukiConfig.from_yaml(args.show)
        print(json.dumps(cfg.to_dict(), indent=2))


def _ensure_utf8_stdio() -> None:
    """Ensure stdout/stderr handle unicode on Windows cp1252 consoles."""
    import io
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
        else:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")  # type: ignore
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")  # type: ignore
    except Exception:
        pass


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))


def _cmd_infer(args: argparse.Namespace) -> None:
    _ensure_utf8_stdio()
    from .config import MiyukiConfig, load_config_or_default
    from .inference.pipeline import run_inference, save_result

    if args.config:
        cfg = MiyukiConfig.from_yaml(args.config)
    else:
        cfg = load_config_or_default()

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    if not prompt:
        print("error: --prompt or --prompt-file required", file=sys.stderr)
        sys.exit(2)

    overrides: dict = {}
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.max_new_tokens is not None:
        overrides["max_new_tokens"] = args.max_new_tokens
    if args.top_p is not None:
        overrides["top_p"] = args.top_p

    # Support loading an adapter checkpoint directly
    if args.adapter:
        from .runtime.transformers_runtime import TransformersRuntime
        runtime = TransformersRuntime(cfg)
        runtime.load()
        runtime.load_adapter_from(args.adapter)
        result_obj = runtime.generate(prompt, **overrides)
        from datetime import datetime, timezone
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": result_obj.prompt,
            "generated": result_obj.generated,
            "full_text": result_obj.full_text,
            "model": result_obj.model_name,
            "config": result_obj.config_snapshot,
            "runtime": result_obj.runtime_info,
            "generation_params": result_obj.generation_params,
            "timings": result_obj.timings,
        }
    else:
        record = run_inference(cfg, prompt, **overrides)

    # Output handling
    text = record["generated"]
    _safe_print("\n--- GENERATED ---\n")
    _safe_print(text)
    _safe_print("\n--- END ---\n")

    if args.output or args.save:
        out = args.output or f"outputs/inference_{record['timestamp'].replace(':','-')}.json"
        save_result(record, out)
        print(f"Saved to {out}")

    if args.json_out:
        _safe_print(json.dumps(record, ensure_ascii=False, indent=2))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from .evaluation.runner import run_evaluation
    from .config import MiyukiConfig, load_config_or_default

    cfg = None
    if args.config:
        cfg = MiyukiConfig.from_yaml(args.config)
    else:
        cfg = load_config_or_default()

    result = run_evaluation(
        model_config=cfg,
        prompts_file=args.prompts,
        output_dir=args.output_dir,
        generation_overrides={"temperature": args.temperature, "max_new_tokens": args.max_new_tokens} if args.temperature else None,
    )
    print(f"Evaluation complete: {result['run_dir']}")
    print(f"  prompts: {result['num_prompts']}, outputs: {result['num_outputs']}")
    print(f"  report: {result['report_path']}")


def _cmd_dataset(args: argparse.Namespace) -> None:
    from .datasets.validate import validate_file
    from .datasets.versioning import compute_manifest
    from pathlib import Path as P

    if args.dataset_cmd == "validate":
        report = validate_file(args.path, format=args.format)
        print(report.summary())
        if args.verbose or not report.passed:
            for e in report.errors[:50]:
                print(f"  ERROR line {e.line}: [{e.code}] {e.message}")
            for w in report.warnings[:50]:
                print(f"  WARN  line {w.line}: [{w.code}] {w.message}")
        if args.output:
            P(args.output).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            print(f"Report written to {args.output}")
        if not report.passed and args.strict:
            sys.exit(1)

    elif args.dataset_cmd == "stats":
        report = validate_file(args.path, format=args.format)
        print(json.dumps(report.stats, indent=2))
        print(f"Total: {report.total}, Valid: {report.valid}, Invalid: {report.invalid}")

    elif args.dataset_cmd == "manifest":
        files = {args.split: args.path}
        manifest = compute_manifest(
            name=args.name,
            version=args.version,
            files=files,
            source=args.source,
            notes=args.notes,
        )
        out = args.output or f"data/manifests/{args.name}-{args.version}.json"
        manifest.to_json(out)
        print(f"Manifest written to {out}")
        print(json.dumps(manifest.model_dump(mode="python"), indent=2))


def _cmd_train(args: argparse.Namespace) -> None:
    from .config import MiyukiConfig
    from .training.train import train

    cfg = MiyukiConfig.from_yaml(args.config) if args.config else None
    # dataset path override
    dataset_path = args.dataset
    train(config=cfg, config_path=args.config, dataset_path=dataset_path, smoke=args.smoke)


def _cmd_experiment(args: argparse.Namespace) -> None:
    from .experiments.tracking import list_experiments, show_experiment
    if args.exp_cmd == "list":
        exps = list_experiments()
        if not exps:
            print("No experiments found.")
        for e in exps:
            print(f"{e['id']}: {e.get('model','?')} | {e.get('dataset','?')} | {e.get('status','?')} | {e.get('created','')}")
    elif args.exp_cmd == "show":
        info = show_experiment(args.id)
        print(json.dumps(info, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="miyuki", description="Miyuki Model Lab CLI")
    sub = p.add_subparsers(dest="command", required=True)

    # hardware
    ph = sub.add_parser("hardware", help="Show hardware report")
    ph.add_argument("--json", action="store_true", help="Output as JSON")
    ph.set_defaults(func=_cmd_hardware)

    # models
    pm = sub.add_parser("models", help="List/show model configs")
    pm.add_argument("--search-dir", default="configs/model")
    pm.add_argument("--list", action="store_true", help="List configs", dest="list")
    pm.add_argument("--show", type=str, help="Show a specific config file")
    # default to list if no flag
    pm.set_defaults(func=_cmd_models)

    # infer
    pi = sub.add_parser("infer", help="Run baseline inference")
    pi.add_argument("--config", type=str, default=None, help="Model config YAML")
    pi.add_argument("--prompt", type=str, default=None)
    pi.add_argument("--prompt-file", type=str, default=None)
    pi.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter checkpoint")
    pi.add_argument("--temperature", type=float, default=None)
    pi.add_argument("--top-p", type=float, default=None)
    pi.add_argument("--max-new-tokens", type=int, default=None)
    pi.add_argument("--save", action="store_true")
    pi.add_argument("--output", type=str, default=None)
    pi.add_argument("--json-out", action="store_true")
    pi.set_defaults(func=_cmd_infer)

    # evaluate
    pe = sub.add_parser("evaluate", help="Run evaluation suite")
    pe.add_argument("--config", type=str, default=None)
    pe.add_argument("--prompts", type=str, default=None, help="Prompts JSONL file")
    pe.add_argument("--output-dir", type=str, default=None)
    pe.add_argument("--temperature", type=float, default=None)
    pe.add_argument("--max-new-tokens", type=int, default=None)
    pe.set_defaults(func=_cmd_evaluate)

    # dataset
    pd = sub.add_parser("dataset", help="Dataset tools")
    dsub = pd.add_subparsers(dest="dataset_cmd", required=True)
    pv = dsub.add_parser("validate", help="Validate a JSONL dataset")
    pv.add_argument("--path", required=True)
    pv.add_argument("--format", default="auto", choices=["auto", "raw", "prompt_completion", "conversational", "story"])
    pv.add_argument("--output", default=None)
    pv.add_argument("--verbose", action="store_true")
    pv.add_argument("--strict", action="store_true")
    ps = dsub.add_parser("stats", help="Show dataset stats")
    ps.add_argument("--path", required=True)
    ps.add_argument("--format", default="auto")
    pmf = dsub.add_parser("manifest", help="Create dataset manifest")
    pmf.add_argument("--path", required=True)
    pmf.add_argument("--split", default="train")
    pmf.add_argument("--name", required=True)
    pmf.add_argument("--version", required=True)
    pmf.add_argument("--source", default=None)
    pmf.add_argument("--notes", default=None)
    pmf.add_argument("--output", default=None)
    pd.set_defaults(func=_cmd_dataset)

    # train
    pt = sub.add_parser("train", help="Run training")
    pt.add_argument("--config", type=str, default="configs/training/smoke.yaml")
    pt.add_argument("--dataset", type=str, default=None)
    pt.add_argument("--smoke", action="store_true", help="Force smoke mode (tiny synthetic dataset)")
    pt.set_defaults(func=_cmd_train)

    # experiment
    px = sub.add_parser("experiment", help="Experiment tracking")
    xsub = px.add_subparsers(dest="exp_cmd", required=True)
    xl = xsub.add_parser("list", help="List experiments")
    xs = xsub.add_parser("show", help="Show experiment")
    xs.add_argument("--id", required=True)
    px.set_defaults(func=_cmd_experiment)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # models default behavior: if no --list or --show, list
    if getattr(args, "command", None) == "models" and not getattr(args, "list", False) and not getattr(args, "show", None):
        args.list = True
    func = getattr(args, "func", None)
    if func:
        func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
