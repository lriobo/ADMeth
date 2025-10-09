import argparse
import steps_preprocess as sp
import steps_evaluate as se
import steps_recscores as sr
import steps_mlmodels as sm
import steps_plots as stp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "cmd",
        choices=["preprocess", "evaluate", "recscores", "mlmodels", "plots", "all"],  
    )
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.cmd == "preprocess":
        sp.run(cfg)
    elif args.cmd == "evaluate":
        se.run(cfg)
    elif args.cmd == "recscores":          
        sr.run(cfg)
    elif args.cmd == "mlmodels": 
        sm.run(cfg)
    elif args.cmd == "plots":
        stp.run(cfg)
    elif args.cmd == "all":
        sp.run(cfg)
        se.run(cfg)
        sr.run(cfg)
        sm.run(cfg)
        stp.run(cfg)
