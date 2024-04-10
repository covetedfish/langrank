from run_all import main as run_main
run_experiment = run_main.callback
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("params", type=str, help="JSON-formatted dictionary of parameters")
    return parser.parse_args()

def main():
    args = parse_args()

    # Print the received JSON string
    print(f"Received JSON string: {args.params}")

    try:
        # Parse the JSON-formatted dictionary
        params = json.loads(args.params)
        if params["ablation"] == "None":
            params["ablation"] = None
        if params["distance"] == "False":
            params["distance"] = False
        if params["distance"] == "True":
            params["distance"] = True
        run_experiment(**params)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {args.params}")
        print(f"JSON decoding error: {e}")
        # Add any additional error handling as needed


if __name__ == "__main__":
    main()

