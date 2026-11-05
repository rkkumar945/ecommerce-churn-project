import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphlib import CycleError

import pytest

from src.pipeline.mini_dag import DAG, Task


def quiet(*_):
    pass


def _noop():
    pass


def test_order_levels_and_determinism(tmp_path):
    dag = DAG([Task('c', _noop, deps=['a', 'b']), Task('a', _noop), Task('b', _noop),
               Task('d', _noop, deps=['c'])], tmp_path)
    assert dag.levels() == [['a', 'b'], ['c'], ['d']]
    assert dag.order() == ['a', 'b', 'c', 'd']


def test_cycle_is_rejected(tmp_path):
    with pytest.raises(CycleError):
        DAG([Task('a', _noop, deps=['b']), Task('b', _noop, deps=['a'])], tmp_path)


def test_unknown_dependency_and_duplicates(tmp_path):
    with pytest.raises(ValueError):
        DAG([Task('a', _noop, deps=['ghost'])], tmp_path)
    with pytest.raises(ValueError):
        DAG([Task('a', _noop), Task('a', _noop)], tmp_path)


class Flaky:
    def __init__(self, fail_times):
        self.calls, self.fail_times = 0, fail_times

    def __call__(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError('boom')


def test_retries(tmp_path):
    f = Flaky(2)
    assert DAG([Task('t', f, retries=2)], tmp_path / 's1').run(log=quiet) == {'t': 'success'}
    assert f.calls == 3
    g = Flaky(2)
    assert DAG([Task('t', g, retries=1)], tmp_path / 's2').run(log=quiet) == {'t': 'failed'}
    assert g.calls == 2


def test_failure_skips_only_downstream(tmp_path):
    def bad():
        raise RuntimeError('x')

    ran = []
    dag = DAG([Task('a', bad), Task('b', lambda: ran.append('b'), deps=['a']),
               Task('c', lambda: ran.append('c'))], tmp_path)
    assert dag.run(log=quiet) == {'a': 'failed', 'b': 'skipped_upstream', 'c': 'success'}
    assert ran == ['c']


def test_cache_invalidation(tmp_path):
    src, out = tmp_path / 'in.txt', tmp_path / 'out.txt'
    src.write_text('v1')
    calls = []

    def build():
        calls.append(1)
        out.write_text(src.read_text().upper())

    dag = DAG([Task('build', build, inputs=[src], outputs=[out])], tmp_path / 'state')
    assert dag.run(log=quiet) == {'build': 'success'}
    assert dag.run(log=quiet) == {'build': 'cached'}
    src.write_text('v2')
    assert dag.run(log=quiet) == {'build': 'success'}      # input badla
    out.unlink()
    assert dag.run(log=quiet) == {'build': 'success'}      # output gayab
    assert dag.run(log=quiet, force=True) == {'build': 'success'}
    assert len(calls) == 4


def test_version_bump_invalidates_cache(tmp_path):
    src, out = tmp_path / 'in.txt', tmp_path / 'out.txt'
    src.write_text('x')

    def build():
        out.write_text('y')

    state = tmp_path / 'state'
    assert DAG([Task('b', build, inputs=[src], outputs=[out], version='1')], state).run(log=quiet) == {'b': 'success'}
    assert DAG([Task('b', build, inputs=[src], outputs=[out], version='1')], state).run(log=quiet) == {'b': 'cached'}
    assert DAG([Task('b', build, inputs=[src], outputs=[out], version='2')], state).run(log=quiet) == {'b': 'success'}






def test_plan_is_a_dry_run(tmp_path):
    src, mid, out = tmp_path / 'in.txt', tmp_path / 'mid.txt', tmp_path / 'out.txt'
    src.write_text('a')
    calls = []

    def t1():
        calls.append('t1')
        mid.write_text(src.read_text())

    def t2():
        calls.append('t2')
        out.write_text(mid.read_text())

    dag = DAG([Task('t1', t1, inputs=[src], outputs=[mid]),
               Task('t2', t2, deps=['t1'], inputs=[mid], outputs=[out])], tmp_path / 'state')
    assert dag.plan() == {'t1': 'would_run', 't2': 'maybe'}
    assert calls == [] and not (tmp_path / 'state').exists()
    dag.run(log=quiet)
    assert dag.plan() == {'t1': 'cached', 't2': 'cached'}
    src.write_text('b')
    assert dag.plan() == {'t1': 'would_run', 't2': 'maybe'}
    assert dag.plan(force=True) == {'t1': 'would_run', 't2': 'would_run'}