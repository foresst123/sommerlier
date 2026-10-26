"""main() must not import a name locally that it also uses outside that import's `if`.

`from x import name` inside a function makes `name` local to the whole function. main() imported
resolve_diarization_backend inside `if not args.skip_bootstrap:` and used it again later, so
`--skip_bootstrap` died with UnboundLocalError. The module-level import is the one to use.
"""
import ast
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")


def test_main_does_not_shadow_module_level_imports():
    tree = ast.parse(SRC)
    module_names = {a.asname or a.name for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    local = {a.asname or a.name for n in ast.walk(main_fn) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "resolve_diarization_backend" in module_names
    assert "resolve_diarization_backend" not in local, "re-imported inside main(): unbound when --skip_bootstrap"
