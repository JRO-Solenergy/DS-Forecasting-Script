from contextlib import contextmanager

@contextmanager
def stage(name: str):
    """Wrap a block so errors say which stage failed."""
    try:
        yield
    except Exception as e:
        print(f"Error during {name}: {type(e).__name__}: {e}")
        raise