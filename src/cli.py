import argparse, yaml
import steps_preprocess as sp
import steps_evaluate as se

def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["preprocess","evaluate","all"])
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.cmd == "preprocess":
        sp.run(cfg)
    elif args.cmd == "evaluate":
        se.run(cfg)
    else:  # all
        sp.run(cfg)
        se.run(cfg)

if __name__ == "__main__":
    main()
