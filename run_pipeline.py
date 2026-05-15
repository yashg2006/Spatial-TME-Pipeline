"""
Master pipeline runner — executes all phases sequentially.
Usage:
    python run_pipeline.py              # run all phases
    python run_pipeline.py --phase 1    # run a specific phase
    python run_pipeline.py --phase 1 2  # run multiple phases
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_phase(n: int):
    if n == 1:
        from src.phase1_data_preparation import run_phase1
        run_phase1()
    elif n == 2:
        from src.phase2_model import run_phase2
        run_phase2()
    elif n == 3:
        from src.phase3_explainability import run_phase3
        run_phase3()
    elif n == 4:
        from src.phase4_validation import run_phase4
        run_phase4()
    elif n == 5:
        from src.phase5_cross_validation import run_phase5
        run_phase5()
    else:
        print(f"Unknown phase: {n}")


def main():
    parser = argparse.ArgumentParser(description="Spatial TME GNN Pipeline")
    parser.add_argument("--phase", nargs="*", type=int,
                        help="Phase(s) to run (1-5). Default: all.")
    args = parser.parse_args()

    phases = args.phase if args.phase else [1, 2, 3, 4, 5]
    print(f"\n--- Running phases: {phases} ---\n")
    for p in phases:
        run_phase(p)
    print("\n*** All requested phases complete. ***")


if __name__ == "__main__":
    main()
