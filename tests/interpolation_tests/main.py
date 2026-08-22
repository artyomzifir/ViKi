import sys
import json
from viki.optimization.interpolation.interpolation import Interpolator


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <input.json> <output.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    interpolator = Interpolator()
    result = interpolator.process(raw_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, fp=f, indent=2)


if __name__ == "__main__":
    main()
