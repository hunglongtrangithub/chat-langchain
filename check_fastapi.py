# check_fastapi.py
try:
    import fastapi

    print("FastAPI is installed.")
except ImportError:
    print("FastAPI is not installed.")
    raise SystemExit(1)
