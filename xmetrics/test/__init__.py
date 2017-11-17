import re
from contextlib import contextmanager

import pytest


@contextmanager
def raises_regex(error, pattern):
    __tracebackhide__ = True  # noqa: F841
    with pytest.raises(error) as excinfo:
        yield
    message = str(excinfo.value)
    if not re.search(pattern, message):
        raise AssertionError('exception %r did not match pattern %r'
                             % (excinfo.value, pattern))
