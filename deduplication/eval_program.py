from contextlib import redirect_stdout
import importlib.util
import timeout_decorator
import io as IO
import resource
import sys
from types import FunctionType
from typing import Any, List, Tuple

def eval_program(code: str, timeout: float = 3.0):
    exc = None
    @timeout_decorator.timeout(timeout)
    def eval_program_impl():
        nonlocal exc
        # Save resource usage limit
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        recursion_limit = sys.getrecursionlimit()
        try:
            with redirect_stdout(IO.StringIO()):
                # Limit resource usage
                resource.setrlimit(resource.RLIMIT_AS, (1 << 32, hard))
                sys.setrecursionlimit(10000)
                exec(code)
        except Exception as err:
            print(err)
            exc = err
        finally:
            sys.setrecursionlimit(recursion_limit)
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    eval_program_impl()
    return exc