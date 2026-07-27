#!/usr/bin/env python3
"""
Wrapper script to run probe pipeline with multiple learning rate and sample size variations.
Calls run_probe_pipeline.py with different parameter combinations for apollorepe and obfact probes.
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path
import argparse

def read_target_layers_and_probe_type_from_metadata(model_checkpoint_path):
    """Read target layers and probe type from model checkpoint metadata.

    This looks for probe checkpoint information in the model's metadata and
    reads the target layers and probe type from the probe checkpoint metadata.
    """
    # Get project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # First, try to read model training metadata to find probe checkpoint
    model_metadata_path = os.path.join(model_checkpoint_path, "model_training_metadata.json")
    probe_checkpoint_path = None

    if os.path.exists(model_metadata_path):
        try:
            with open(model_metadata_path, "r") as f:
                model_metadata = json.load(f)
                probe_checkpoint_path = model_metadata.get("probe_checkpoints")
        except Exception as e:
            print(f"Warning: Could not read model training metadata: {e}")

    # If we found probe checkpoint path from model metadata, read its metadata
    if probe_checkpoint_path and os.path.exists(os.path.join(project_root, probe_checkpoint_path)):
        metadata_path = os.path.join(project_root, probe_checkpoint_path, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    if "target_layers" in metadata and "probe_type" in metadata:
                        # Convert list of ints to comma-separated string
                        target_layers = ",".join(str(l) for l in metadata["target_layers"])
                        probe_type = metadata["probe_type"]
                        print(f"Read from probe metadata: target_layers={target_layers}, probe_type={probe_type}")
                        return target_layers, probe_type
            except Exception as e:
                print(f"Warning: Could not read probe metadata: {e}")

    # Fallback: use default values
    print("Warning: Could not read target layers and probe type from metadata, using defaults")
    return "12", "logistic"

def run_probe_pipeline_variation(model_checkpoint, variation_name, target_layers, probe_type,
                                apollo_lr=None, obfact_lr=None, deception_lr=None, merged_harmful_lr=None,
                                apollo_limit=None, deception_limit=None, obfact_limit=None, merged_harmful_limit=None,
                                output_base=None, skip_apollo=False, skip_obfact=False,
                                skip_deception=False, skip_merged_harmful=False, skip_harmful=False, custom_args=None):
    """Run the probe pipeline with specific parameters."""

    print(f"\n{'='*60}")
    print(f"RUNNING VARIATION: {variation_name}")
    if apollo_lr: print(f"Apollo LR: {apollo_lr}, limit: {apollo_limit}")
    if obfact_lr: print(f"Obfact LR: {obfact_lr}, limit: {obfact_limit}")
    if deception_lr: print(f"Deception LR: {deception_lr}, limit: {deception_limit}")
    if merged_harmful_lr: print(f"Merged Harmful LR: {merged_harmful_lr}, limit: {merged_harmful_limit}")
    print(f"Target layers: {target_layers}, Probe type: {probe_type}")
    print(f"{'='*60}")

    # Build command
    cmd = [
        sys.executable,
        "experiments/pipeline/run_probe_pipeline.py",
        model_checkpoint,
        "--target-layers", str(target_layers),
        "--probe-type", str(probe_type)
    ]

    # Add probe-specific parameters
    if apollo_lr:
        cmd.extend(["--apollo-lr", str(apollo_lr)])
    if apollo_limit:
        cmd.extend(["--apollo-train-limit", str(apollo_limit)])
    if obfact_lr:
        cmd.extend(["--obfact-lr", str(obfact_lr)])
    if obfact_limit:
        cmd.extend(["--obfact-train-limit", str(obfact_limit)])
    if deception_lr:
        cmd.extend(["--deception-lr", str(deception_lr)])
    if deception_limit:
        cmd.extend(["--deception-train-limit", str(deception_limit)])
    if merged_harmful_lr:
        cmd.extend(["--merged-harmful-lr", str(merged_harmful_lr)])
    if merged_harmful_limit:
        cmd.extend(["--merged-harmful-train-limit", str(merged_harmful_limit)])

    # Add output directory if provided
    if output_base:
        cmd.extend(["--output-dir", output_base])

    # Configure which probe types to skip based on parameters
    if skip_merged_harmful:
        cmd.append("--skip-merged")
    if skip_deception:
        cmd.append("--skip-deception")
    if skip_apollo:
        cmd.append("--skip-apollo-repe")
    if skip_obfact:
        cmd.extend(["--skip-obfact-harmful"])
    if skip_harmful:
        cmd.append("--skip-harmful")

    # Add custom arguments if provided
    if custom_args:
        cmd.extend(custom_args)

    # Add medium test mode by default
    cmd.append("--medium-test")

    print(f"Command: {' '.join(cmd)}")

    # Create log file for this variation (if output_base is provided)
    if output_base:
        log_dir = Path(output_base) / "probe_variation_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{variation_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_log.txt"
    else:
        log_file = None

    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Write logs if we have a log file
        if log_file:
            with open(log_file, 'w') as f:
                f.write(f"STDOUT:\n{result.stdout}\n\n")
                f.write(f"STDERR:\n{result.stderr}\n")

        # Always print key lines to console
        for line in result.stdout.split('\n'):
            if any(keyword in line for keyword in ['✓', '✗', 'completed', 'failed', 'ERROR', 'Success']):
                print(line)

        print(f"✓ Variation {variation_name} completed successfully")
        if log_file:
            print(f"  Log saved to: {log_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Variation {variation_name} failed with exit code {e.returncode}")
        if log_file:
            with open(log_file, 'w') as f:
                f.write(f"Command failed with exit code {e.returncode}\n")
                f.write(f"STDOUT:\n{e.stdout}\n\n")
                f.write(f"STDERR:\n{e.stderr}\n")
        else:
            print(f"STDOUT:\n{e.stdout}")
            print(f"STDERR:\n{e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run probe pipeline with multiple variations")
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=None,
        help="Base output directory for results (optional)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run in quick test mode"
    )
    parser.add_argument(
        "--full-mode",
        action="store_true",
        help="Run in full mode"
    )
    parser.add_argument(
        "--deception-and-apollo-only",
        action="store_true"
    )

    args = parser.parse_args()

    # Read target layers and probe type from model checkpoint metadata
    target_layers, probe_type = read_target_layers_and_probe_type_from_metadata(args.model_checkpoint)



    # Define the 4 variations from run_full_pipeline.py
    variations = [
        {
            "name": "run1_apollo_deception_merged_obfact",
            "description": "Apollo (LR 0.001, limit 1000), Deception (LR 0.001, limit 1000), Merged harm (LR 0.001, limit 200), Obfact harm (LR 8e-05, limit 1000)",
            "apollo_lr": 0.001,
            "apollo_limit": 1000,
            "deception_lr": 0.001,
            "deception_limit": 1000,
            "merged_harmful_lr": 0.001,
            "merged_harmful_limit": 200,
            "obfact_lr": 8e-05,
            "obfact_limit": 1000,
            "custom_args": ["--skip-harmful", "--skip-training-concepts"] + (["--skip-obfact-harmful", "--skip-merged"] if args.deception_and_apollo_only else [])
        },
        {
            "name": "run2_apollo_merged_obfact_skip_deception",
            "description": "Apollo (LR 8e-05, limit 1000), Merged harm (LR 8e-05, limit 200), Obfact harm (LR 0.001, limit 1000), skip deception",
            "apollo_lr": 8e-05,
            "apollo_limit": 1000,
            "merged_harmful_lr": 8e-05,
            "merged_harmful_limit": 200,
            "obfact_lr": 0.001,
            "obfact_limit": 1000,
            "skip_deception": True,
            "custom_args": ["--skip-deception", "--skip-harmful", "--skip-training-concepts"] + (["--skip-obfact-harmful", "--skip-merged"] if args.deception_and_apollo_only else [])
        },
        {
            "name": "run3_merged_obfact_skip_apollo_deception",
            "description": "Merged harm (LR 0.001, limit 1000), Obfact harm (LR 4e-05, limit 2000), skip Apollo and deception",
            "merged_harmful_lr": 0.001,
            "merged_harmful_limit": 1000,
            "obfact_lr": 4e-05,
            "obfact_limit": 2000,
            "skip_apollo": True,
            "skip_deception": True,
            "custom_args": ["--skip-apollo-repe", "--skip-deception", "--skip-harmful", "--skip-training-concepts"]
        },
        {
            "name": "run4_merged_only",
            "description": "Merged harm only (LR 8e-05, limit 1000), skip Apollo, deception, and obfact",
            "merged_harmful_lr": 8e-05,
            "merged_harmful_limit": 1000,
            "skip_apollo": True,
            "skip_deception": True,
            "skip_obfact": True,
            "custom_args": ["--skip-apollo-repe", "--skip-deception", "--skip-obfact-harmful", "--skip-harmful", "--skip-training-concepts"]
        },
    ]

    if args.deception_and_apollo_only:
        variations = variations[:2]

    # Create output directory if specified
    if args.output_base:
        output_base = Path(args.output_base)
        output_base.mkdir(parents=True, exist_ok=True)
    else:
        args.output_base = None

    print(f"\n{'='*80}")
    print("PROBE PIPELINE VARIATIONS RUNNER")
    print(f"{'='*80}")
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"Target layers: {target_layers}")
    print(f"Probe type: {probe_type}")
    print(f"Output directory: {args.output_base}")
    print(f"Number of variations: {len(variations)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    results = []

    for i, var in enumerate(variations, 1):
        print(f"\n{'='*80}")
        print(f"VARIATION {i}/{len(variations)}: {var['name']}")
        print(f"Description: {var['description']}")
        print(f"{'='*80}")

        success = run_probe_pipeline_variation(
            args.model_checkpoint,
            var["name"],
            target_layers,
            probe_type,
            apollo_lr=var.get("apollo_lr"),
            obfact_lr=var.get("obfact_lr"),
            deception_lr=var.get("deception_lr"),
            merged_harmful_lr=var.get("merged_harmful_lr"),
            apollo_limit=var.get("apollo_limit"),
            deception_limit=var.get("deception_limit"),
            obfact_limit=var.get("obfact_limit"),
            merged_harmful_limit=var.get("merged_harmful_limit"),
            output_base=args.output_base,
            skip_apollo=var.get("skip_apollo", False),
            skip_obfact=var.get("skip_obfact", False),
            skip_deception=var.get("skip_deception", False),
            skip_merged_harmful=var.get("skip_merged_harmful", False),
            skip_harmful=var.get("skip_harmful", False),
            custom_args=var.get("custom_args", [])
        )

        result_info = {
            "variation": var["name"],
            "description": var["description"],
            "success": success
        }
        # Add probe-specific info to results
        for key in ["apollo_lr", "obfact_lr", "deception_lr", "merged_harmful_lr",
                   "apollo_limit", "obfact_limit", "deception_limit", "merged_harmful_limit"]:
            if key in var:
                result_info[key] = var[key]
        results.append(result_info)

    # Save summary if output directory is specified
    if args.output_base:
        summary = {
            "model_checkpoint": args.model_checkpoint,
            "target_layers": target_layers,
            "probe_type": probe_type,
            "output_directory": str(args.output_base),
            "timestamp": datetime.now().isoformat(),
            "variations": variations,
            "results": results
        }

        summary_file = output_base / "variations_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")

    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    successful = sum(1 for r in results if r["success"])
    print(f"Total variations run: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")

    print("\nDetailed Results:")
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['variation']}")
        print(f"     Description: {r['description']}")

    print(f"{'='*80}")

    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
