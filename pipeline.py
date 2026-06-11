import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STEPS = {
    "handwriting": "build_handwritten_words.py",
    "convert": "convert_to_donut.py",
    "synthetic": "generate_synthetic.py",
    "merge": "merge_dataset.py",
    "train": "train.py",
    "evaluate": "evaluate.py",
}

PIPELINE_ORDER = [
    "handwriting",
    "convert",
    "synthetic",
    "merge",
    "train",
    "evaluate",
]


def run_step(step_name):
    script = STEPS[step_name]
    script_path = BASE_DIR / script

    print("\n" + "=" * 70)
    print(f"RUNNING: {step_name}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR
    )

    if result.returncode != 0:
        print(f"\nFAILED: {step_name}")
        sys.exit(result.returncode)

    print(f"\nCOMPLETED: {step_name}")


def main():
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python pipeline.py all")
        print("  python pipeline.py train")
        print("  python pipeline.py synthetic")
        return

    command = sys.argv[1].lower()

    if command == "all":
        for step in PIPELINE_ORDER:
            run_step(step)

    elif command in STEPS:
        run_step(command)

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
