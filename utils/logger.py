from datetime import datetime


def ts():
    return datetime.now().strftime("%H:%M:%S")

def header(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")

def stage(name: str, msg: str):
    print(f"\n[{ts()}] [{name}] {msg}")

def sub(msg: str):
    print(f"            → {msg}")

def info(label: str, value: str):
    print(f"            {label:<28} {value}")

def divider():
    print(f"  {'-' * 61}")

def warn(msg: str):
    print(f"[{ts()}] ⚠  {msg}")

def ok(msg: str):
    print(f"[{ts()}] ✓  {msg}")