from __future__ import annotations

import argparse
import json
from typing import Sequence

from qwen3_omni_pretrain.architecture.profiles import ArchitectureProfile
from qwen3_omni_pretrain.profiles.registry import (
    ProfileBuildRequest,
    get_profile_factory,
    parse_profile,
)


def _add_request_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_inspection_options: bool,
) -> None:
    parser.add_argument(
        "--profile",
        required=True,
        help="Exact architecture profile identifier.",
    )
    parser.add_argument(
        "--config-or-checkpoint",
        required=True,
        help="Local config/checkpoint path or a reference model ID.",
    )
    if include_inspection_options:
        parser.add_argument(
            "--dtype",
            help="Optional build dtype (legacy inspection only).",
        )
        parser.add_argument(
            "--tokenizer",
            help="Optional tokenizer source (legacy inspection only).",
        )
        parser.add_argument(
            "--capability",
            action="append",
            default=[],
            dest="requested_capabilities",
            help="Capability that the caller requires; repeat as needed.",
        )
        parser.add_argument(
            "--allow-network",
            action="store_true",
            help=(
                "Permit the legacy tokenizer lookup to use the network; "
                "offline is the default."
            ),
        )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print stable machine-readable JSON.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m qwen3_omni_pretrain.cli_profile",
        description=(
            "Validate or inspect an architecture profile without loading "
            "checkpoint weights."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration and print its profile manifest.",
    )
    _add_request_arguments(
        validate_parser,
        include_inspection_options=False,
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Build a metadata-only artifact and print its architecture.",
    )
    _add_request_arguments(
        inspect_parser,
        include_inspection_options=True,
    )
    return parser


def _validate_cli_options(
    args: argparse.Namespace,
    profile: ArchitectureProfile,
) -> None:
    if args.command != "inspect":
        return
    if (
        args.dtype is not None
        and profile is not ArchitectureProfile.LEGACY_PROTOTYPE
    ):
        raise ValueError(
            "--dtype is only valid for legacy_prototype inspection"
        )
    if (
        args.tokenizer is not None
        and profile is not ArchitectureProfile.LEGACY_PROTOTYPE
    ):
        raise ValueError(
            "--tokenizer is only valid for legacy_prototype inspection"
        )
    if args.allow_network and not (
        profile is ArchitectureProfile.LEGACY_PROTOTYPE
        and args.tokenizer is not None
    ):
        raise ValueError(
            "--allow-network requires legacy_prototype inspection "
            "with --tokenizer"
        )


def _request_from_args(
    args: argparse.Namespace,
    profile: ArchitectureProfile,
) -> ProfileBuildRequest:
    device = (
        "meta"
        if (
            args.command == "inspect"
            and profile is ArchitectureProfile.LEGACY_PROTOTYPE
        )
        else None
    )
    return ProfileBuildRequest(
        profile=profile,
        config_or_checkpoint=args.config_or_checkpoint,
        tokenizer=getattr(args, "tokenizer", None),
        local_files_only=not getattr(args, "allow_network", False),
        dtype=getattr(args, "dtype", None),
        device=device,
        requested_capabilities=tuple(
            getattr(args, "requested_capabilities", ())
        ),
    )


def _print_payload(
    payload: dict[str, object],
    *,
    as_json: bool,
) -> None:
    if as_json:
        print(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        profile = parse_profile(args.profile)
        _validate_cli_options(args, profile)
        factory = get_profile_factory(profile)
        request = _request_from_args(args, profile)
        if args.command == "validate":
            payload = factory.validate(request).to_dict()
        else:
            result = factory.build(request)
            payload = {
                "artifact_type": type(result.artifact).__name__,
                "manifest": result.manifest.to_dict(),
                "architecture_summary": (
                    result.architecture_summary.to_dict()
                ),
            }
    except Exception as exc:
        parser.error(str(exc))

    _print_payload(payload, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
