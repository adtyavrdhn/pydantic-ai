"""Measurement only: name the test that created each event loop or sqlite connection which is garbage collected unclosed.

Loaded with `-p leakplugin`. Appends one JSON line per leak to $LEAK_DIR/<pid>.jsonl.
"""
import asyncio.base_events as base_events
import gc
import json
import os
import sqlite3
import traceback
from functools import wraps
from pathlib import Path

DIR = Path(os.environ['LEAK_DIR'])
DIR.mkdir(parents=True, exist_ok=True)


def current_test():
    return os.environ.get('PYTEST_CURRENT_TEST', '<outside a test>')


def where():
    frames = [f for f in traceback.extract_stack()[:-2] if 'leakplugin' not in f.filename]
    keep = [f for f in frames if 'site-packages' not in f.filename and '/lib/python' not in f.filename][-4:]
    return [f'{f.filename.split("pydantic-ai/")[-1]}:{f.lineno}:{f.name}' for f in keep]


def report(kind, created_in, stack):
    with open(DIR / f'{os.getpid()}.jsonl', 'a') as out:
        out.write(json.dumps(dict(kind=kind, created_in=created_in, collected_in=current_test(), stack=stack)) + '\n')


original_init = base_events.BaseEventLoop.__init__
original_del = base_events.BaseEventLoop.__del__


@wraps(original_init)
def loop_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self._leak_origin = (current_test(), where())


def loop_del(self, *args, **kwargs):
    if not self.is_closed():
        origin = getattr(self, '_leak_origin', ('?', []))
        report('event_loop', origin[0], origin[1])
    original_del(self, *args, **kwargs)


base_events.BaseEventLoop.__init__ = loop_init
base_events.BaseEventLoop.__del__ = loop_del


class TrackedConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._leak_origin = (current_test(), where())
        self._leak_closed = False

    def close(self):
        self._leak_closed = True
        super().close()

    def __del__(self):
        if not getattr(self, '_leak_closed', True):
            report('sqlite', *self._leak_origin)


original_connect = sqlite3.connect


@wraps(original_connect)
def tracked_connect(*args, **kwargs):
    kwargs.setdefault('factory', TrackedConnection)
    return original_connect(*args, **kwargs)


sqlite3.connect = tracked_connect
sqlite3.dbapi2.connect = tracked_connect


def pytest_sessionfinish(session, exitstatus):
    gc.collect()
