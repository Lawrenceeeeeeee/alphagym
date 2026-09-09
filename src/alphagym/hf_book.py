"""Causal reconstruction of archived L2 messages into clock-time research samples."""
from __future__ import annotations

import bisect
import json
import math
from collections import Counter

import pandas as pd

from alphagym.equity_data import DataContractError

SAMPLER_VERSION = 2
LEVELS = (1, 2, 3, 5, 10, 20, 50, 100)
PRESSURE_LEVELS = (5, 10, 20, 50)

class BookSide:
    def __init__(self, bid=False):
        self.sign = -1 if bid else 1
        self.prices = []
        self.sizes = {}

    def update(self, levels):
        seen = set()
        for level in levels:
            price, size = float(level[0]), float(level[1])
            if not math.isfinite(price + size) or price <= 0 or size < 0:
                raise DataContractError("Invalid order-book price or size")
            key = self.sign * price
            if key in seen:
                raise DataContractError("Duplicate price in order-book message")
            seen.add(key)
            if size == 0:
                if key in self.sizes:
                    self.prices.pop(bisect.bisect_left(self.prices, key))
                    del self.sizes[key]
            else:
                if key not in self.sizes:
                    bisect.insort(self.prices, key)
                self.sizes[key] = size

    def top(self, n):
        return [(self.sign * p, self.sizes[p]) for p in self.prices[:n]]


class BookSampler:
    """Emit grid observations BEFORE applying messages later than the grid.

    Equal-timestamp messages are consumed together. Samples use only exchange
    timestamps; archive receipt timestamps/sequence IDs may be unavailable.
    No forward fill across a day boundary, crossed book or stale observation.
    """

    def __init__(self, interval_ms=1000, max_stale_ms=2000):
        if interval_ms <= 0 or max_stale_ms <= 0:
            raise DataContractError("Sampling interval and maximum age must be positive")
        self.interval = interval_ms
        self.max_stale = max_stale_ms
        self.bid, self.ask = BookSide(True), BookSide()
        self.last_ts = None
        self.last_row = -1
        self.next_grid = None
        self.ready = False
        self.segment = 0
        self.ofi = 0.0
        self.events = 0
        self.stats = Counter()
        self.rows = []

    def _bbo(self):
        if not self.bid.prices or not self.ask.prices:
            return None
        b, a = self.bid.top(1)[0], self.ask.top(1)[0]
        return (*b, *a) if b[0] < a[0] else None

    def _sample(self, ts):
        bbo = self._bbo()
        valid = self.ready and bbo is not None and ts - self.last_ts <= self.max_stale
        if not valid:
            self.stats['invalid_grid'] += 1
            self.ofi, self.events = 0.0, 0
            return
        bp, bq, ap, aq = bbo
        row = {'ts': ts, 'book_ts': self.last_ts, 'segment': self.segment,
               'bid': bp, 'ask': ap, 'bid_size': bq, 'ask_size': aq,
               'mid': (bp + ap) / 2, 'ofi': self.ofi, 'events': self.events}
        for n in LEVELS:
            b, a = self.bid.top(n), self.ask.top(n)
            row[f'bdepth_{n}'] = sum(x[1] for x in b) if len(b) == n else float('nan')
            row[f'adepth_{n}'] = sum(x[1] for x in a) if len(a) == n else float('nan')
            row[f'bnotional_{n}'] = sum(x[0]*x[1] for x in b) if len(b) == n else float('nan')
            row[f'anotional_{n}'] = sum(x[0]*x[1] for x in a) if len(a) == n else float('nan')
            row[f'bid_distance_{n}'] = bp-b[-1][0] if len(b) == n else float('nan')
            row[f'ask_distance_{n}'] = a[-1][0]-ap if len(a) == n else float('nan')
            if len(b) == n and len(a) == n:
                for label, weights in self._pressure_weights(n, b, a, bp, ap).items():
                    row[f'bpressure_{label}_{n}'] = sum(q*w for (_, q), w in zip(b, weights[0]))
                    row[f'apressure_{label}_{n}'] = sum(q*w for (_, q), w in zip(a, weights[1]))
        self.rows.append(row)
        self.ofi, self.events = 0.0, 0

    @staticmethod
    def _pressure_weights(n, bids, asks, bp, ap):
        if n not in PRESSURE_LEVELS:
            return {}
        levels = list(range(1, n+1))
        return {
            'inverse_level': ([1/x for x in levels], [1/x for x in levels]),
            'linear': ([(n+1-x)/n for x in levels], [(n+1-x)/n for x in levels]),
            'exp2': ([2**(-(x-1)/2) for x in levels], [2**(-(x-1)/2) for x in levels]),
            'inverse_distance': (
                [1/(1+max(0, bp-p)) for p, _ in bids],
                [1/(1+max(0, p-ap)) for p, _ in asks],
            ),
        }

    def consume(self, records):
        for record in records:
            ts, source_row = int(record['ts']), int(record['source_row'])
            if source_row != self.last_row + 1:
                raise DataContractError("Order-book source rows are missing or out of order")
            if self.last_ts is not None and ts < self.last_ts:
                raise DataContractError("Order-book timestamps move backwards")
            if self.next_grid is None:
                self.next_grid = (ts // self.interval + 1) * self.interval
            while self.next_grid < ts:
                self._sample(self.next_grid)
                self.next_grid += self.interval
            previous = self._bbo() if self.ready else None
            action = record['action']
            if action == 'snapshot':
                self.bid, self.ask = BookSide(True), BookSide()
                self.ready = True
                self.segment += 1
                self.ofi = 0.0
                previous = None
            elif action != 'update':
                raise DataContractError(f"Unknown order-book action: {action}")
            if not self.ready:
                raise DataContractError("Order-book update precedes initial snapshot")
            for side, target in [('bids', self.bid), ('asks', self.ask)]:
                levels = record[side]
                target.update(json.loads(levels) if isinstance(levels, str) else levels)
            current = self._bbo()
            if current is None:
                self.stats['crossed_or_empty'] += 1
                self.segment += 1
                self.ofi = 0.0
            elif previous is not None:
                bp, bq, ap, aq = current
                pb, pq, pa, pqa = previous
                self.ofi += (bq if bp >= pb else 0) - (pq if bp <= pb else 0)
                self.ofi += -(aq if ap <= pa else 0) + (pqa if ap >= pa else 0)
            self.last_ts, self.last_row = ts, source_row
            self.events += 1
            self.stats[action] += 1
            self.stats['messages'] += 1

    def finish(self):
        # Never invent observations beyond the final received message.
        if self.last_ts is not None and self.next_grid == self.last_ts:
            self._sample(self.next_grid)
            self.next_grid += self.interval
        return pd.DataFrame(self.rows), dict(self.stats)


def build_samples(root, *, start, end):
    """Persist one atomic day at a time; reuse only the same raw data version."""
    from alphagym import storage_io
    from alphagym.config import resolve_root

    root = resolve_root(root)
    store = storage_io.store_for(root)
    dates = pd.date_range(start, end).strftime('%Y-%m-%d')
    if not len(dates):
        raise DataContractError('Empty high-frequency date range')
    results = []
    for date in dates:
        source = f'crypto/okx/order_book/BTC-USDT/400/{date}.parquet'
        target = source.replace('order_book/', 'hf_samples_v1/')
        audit = target.removesuffix('.parquet')+'.json'
        meta = store.manifest(source)
        if meta is None:
            raise DataContractError(f'Missing raw order-book archive: {date}')
        cached = store.manifest(target)
        if cached and store.manifest(audit):
            quality = json.loads(store.read_blob(audit))
            if (quality.get('source_version') == meta['version']
                    and quality.get('sampler_version') == SAMPLER_VERSION):
                results.append(quality)
                continue
        sampler = BookSampler()
        for block in store.iter_batches(source, as_of=meta['version']):
            sampler.consume(block.to_pylist())
        frame, quality = sampler.finish()
        if len(frame) == 0:
            raise DataContractError(f'No valid order-book samples: {date}')
        if quality['messages'] != meta['rows']:
            raise DataContractError(f'Archive row count mismatch: {date}')
        day_start = pd.Timestamp(date, tz='UTC').value//1_000_000
        if not frame.ts.between(day_start, day_start+86400000-1).all():
            raise DataContractError('Archive includes samples outside its UTC source date')
        quality.update(source=source, source_version=meta['version'], samples=len(frame),
                       sampler_version=SAMPLER_VERSION)
        with store.batch() as batch:
            batch.frame(target, frame, keys=('ts',))
            batch.blob(audit, json.dumps(quality).encode())
        results.append(quality)
    return {'ok': True, 'days': len(results), 'quality': results}
