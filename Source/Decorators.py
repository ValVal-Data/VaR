from functools import wraps
import time as ti
from typing import Callable,TypeVar

F=TypeVar("F",bound=Callable[...,object])#Type function for decorator

def timeit(func:F)->F:
    """
    Decorator that measures how long a function take to execute and print the duration in seconds.
    
    Parameters:
        func (Callable[...,object])

    Returns:
        func (Callable[...,object])
    """
    @wraps(func)
    def wrapper(*args,**kwargs):
        start=ti.perf_counter()
        result=func(*args,**kwargs)
        end=ti.perf_counter()
        print(f"{func.__name__}: {end-start:.6f}s")
        return result
    return wrapper