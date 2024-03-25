from pathlib import Path
import functools

import matlab.engine


@functools.cache
def matlab_engine():
    """
    only cached because used by Pytest in multiple tests
    """

    return matlab.engine.start_matlab("-nojvm")
