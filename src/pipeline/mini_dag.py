"""Mini DAG runner (Day 93): orchestration ke concepts samajhne ke liye, Airflow/Prefect ka replacement nahi.
Dependencies, retries, 'inputs badle nahi to skip' (caching) aur failure propagation."""
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from pathlib import Path


@dataclass
class Task:
    name: str
    fn: Callable[[], None]
    deps: list[str] = field(default_factory=list)
    inputs: list[Path] = field(default_factory=list)     # inke content ka hash 'fresh ya stale' tay karta hai
    outputs: list[Path] = field(default_factory=list)    # sab maujood hon tabhi cached mana jaata hai
    retries: int = 0
    retry_delay: float = 0.0
    version: str = '1'                                   # logic badla to bump karo (code ka hash nahi liya jaata)


def _hash_files(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        h.update(p.name.encode())
        h.update(p.read_bytes() if p.exists() else b'<missing>')
    return h.hexdigest()


class DAG:
    def __init__(self, tasks: list[Task], state_dir):
        names = [t.name for t in tasks]
        if len(set(names)) != len(names):
            raise ValueError('duplicate task names')
        self.tasks = {t.name: t for t in tasks}
        for t in tasks:
            for d in t.deps:
                if d not in self.tasks:
                    raise ValueError(f'{t.name} depends on unknown task {d}')
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._sorter().prepare()                         # cycle ho to yahin CycleError

    def _sorter(self) -> TopologicalSorter:
        return TopologicalSorter({n: set(t.deps) for n, t in self.tasks.items()})

    def levels(self) -> list[list[str]]:
        """Har level ke tasks ek saath chal sakte hain (parallelism ka idea)."""
        ts = self._sorter()
        ts.prepare()
        out = []
        while ts.is_active():
            ready = sorted(ts.get_ready())
            out.append(ready)
            ts.done(*ready)
        return out

    def order(self) -> list[str]:
        return [n for level in self.levels() for n in level]

    def _fingerprint(self, t: Task) -> str:
        return hashlib.sha256(f'{t.version}|{_hash_files(t.inputs)}'.encode()).hexdigest()

    def _is_fresh(self, t: Task, fp: str) -> bool:
        state = self.state_dir / f'{t.name}.json'
        if not t.outputs or not state.exists():          # bina outputs wala task hamesha chalega
            return False
        if json.loads(state.read_text()).get('fingerprint') != fp:
            return False
        return all(Path(p).exists() for p in t.outputs)

    def _run_with_retries(self, t: Task, log) -> str:
        for attempt in range(t.retries + 1):
            t0 = time.perf_counter()
            try:
                t.fn()
                log(f'[{t.name}] success in {time.perf_counter() - t0:.2f}s (attempt {attempt + 1})')
                return 'success'
            except Exception as e:  # noqa: BLE001
                log(f'[{t.name}] attempt {attempt + 1} failed: {e!r}')
                if attempt < t.retries:
                    time.sleep(t.retry_delay)
        return 'failed'

    def run(self, force: bool = False, log=print) -> dict[str, str]:
        status: dict[str, str] = {}
        for name in self.order():
            t = self.tasks[name]
            blocked = [d for d in t.deps if status.get(d) in ('failed', 'skipped_upstream')]
            if blocked:
                status[name] = 'skipped_upstream'
                log(f'[{name}] SKIPPED (upstream problem: {blocked})')
                continue
            fp = self._fingerprint(t)
            if not force and self._is_fresh(t, fp):
                status[name] = 'cached'
                log(f'[{name}] cached (inputs unchanged, outputs present)')
                continue
            status[name] = self._run_with_retries(t, log)
            if status[name] == 'success':
                (self.state_dir / f'{name}.json').write_text(json.dumps({'fingerprint': fp}))
        return status