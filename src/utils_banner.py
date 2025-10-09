from termcolor import colored
from datetime import datetime

def print_banner(step: str, project: str = None, emoji: str = "🧮"):
    """
    Clean, centered banner for ADMeth pipeline steps.
    """
    line = "─" * 50
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(colored(line, "cyan"))

    # Centrar el título "⚗️  ADMeth"
    title = f"{emoji}  ADMeth  {emoji}"
    centered_title = title.center(50)
    print(colored(centered_title, "cyan", attrs=["bold"]))

    # Mostrar proyecto y step
    if project:
        print(colored(f"📁 Project: {project}", "green"))
    print(colored(f"🧩 Step:    {step}", "yellow"))
    ts_line = f"🕒 {now}"
    print(colored(ts_line, "magenta"))
    
    print(colored(line, "cyan"))
    print()
