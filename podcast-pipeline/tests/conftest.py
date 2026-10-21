import os

# Tests run main.py's setup section in subprocesses; they must never build environments.
os.environ.setdefault("SOMMELIER_SKIP_BOOTSTRAP", "1")
