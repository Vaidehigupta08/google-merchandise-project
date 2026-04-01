"""
main.py
=======
Module 4 — Full Pipeline Entry Point

Runs all 3 steps in sequence:
    Step 1: Build trajectories from Module 2+3 outputs
    Step 2: Train CFM model
    Step 3: Generate predictions

Usage:
    python main.py              # full pipeline
    python main.py --skip-train # skip training, only predict (needs existing model)
"""

import os
import sys
import argparse

# Make sure imports work from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Module 4 CFM Pipeline")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load existing model for prediction")
    parser.add_argument("--epochs",    type=int,   default=300)
    parser.add_argument("--batch-size",type=int,   default=64)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--steps",     type=int,   default=50,
                        help="ODE integration steps for prediction")
    parser.add_argument("--integration", type=str, default="rk4",
                        choices=["euler", "rk4"])
    args = parser.parse_args()

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE, "module4_Flowboost", "outputs", "cfm_model.pt")

    # ── Step 1: Build trajectories ─────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Building trajectories")
    print("=" * 60)
    from trajectory_builder import build_trajectories
    build_trajectories()

    # ── Step 2: Train ──────────────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("STEP 2: Training CFM model")
        print("=" * 60)
        from train import train
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    else:
        if not os.path.exists(MODEL_PATH):
            print("❌ --skip-train used but no model found at:", MODEL_PATH)
            sys.exit(1)
        print("\n⏭️  Skipping training, using existing model.")

    # ── Step 3: Predict ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Generating predictions")
    print("=" * 60)
    from predict import run_predictions
    run_predictions(integration=args.integration, steps=args.steps)

    print("\n" + "=" * 60)
    print("✅ MODULE 4 COMPLETE")
    print("   Output: module4_Flowboost/outputs/predictions.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
