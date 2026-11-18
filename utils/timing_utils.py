import time
from functools import wraps

def timing(func):
    """
    A decorator to record the computation time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        
        start = time.time()
        result = func(*args, **kwargs)
        
        wrapper.last_time = time.time() - start
        return result
    wrapper.last_time = None
    return wrapper