"""Progress bars for terminals and bounded-frequency progress updates in job logs."""
import os
import sys
from tqdm.auto import tqdm


def progress(iterable=None, **kwargs):
    kwargs.setdefault('disable', os.environ.get('WISER_PROGRESS', '1') == '0')
    kwargs.setdefault('mininterval', 1.0 if sys.stderr.isatty() else 30.0)
    kwargs.setdefault('dynamic_ncols', True)
    kwargs.setdefault('unit', 'batch')
    return tqdm(iterable, **kwargs)
