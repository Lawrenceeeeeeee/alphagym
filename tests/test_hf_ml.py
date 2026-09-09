from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphagym.equity_data import DataContractError
from alphagym.hf_ml import _baseline_predictions, _fit_edge_calibration, load_ml_spec


def test_ml_spec_rejects_test_cost_selection_omissions(tmp_path):
    path = tmp_path/'ml.yaml'
    path.write_text('source_report_id: hf-0123456789abcdef\nworkers: 0\n', encoding='utf-8')
    with pytest.raises(DataContractError, match='Workers'):
        load_ml_spec(path)
    path.write_text(
        'source_report_id: hf-0123456789abcdef\n'
        'fees_bps_per_side: [0, 1]\nselection_fee_bps_per_side: 5\n', encoding='utf-8')
    with pytest.raises(DataContractError, match='Selection fee'):
        load_ml_spec(path)


def test_baseline_direction_and_ic_weights_are_fit_from_training_only():
    names = ['positive', 'negative', 'noise']
    x_train = np.column_stack([
        np.arange(100), -np.arange(100), np.tile([0, 1], 50),
    ]).astype(float)
    y_train = np.arange(100, dtype=float)
    x_all = np.vstack([x_train, [[200, -200, 1]]])
    equal, ic = _baseline_predictions('equal_signed', x_train, y_train, x_all, names)
    weighted, _ = _baseline_predictions('ic_weighted', x_train, y_train, x_all, names)
    assert ic.positive > .99
    assert ic.negative < -.99
    assert equal[-1] > equal[0]
    assert weighted[-1] > weighted[0]
    assert isinstance(pd.Series(weighted), pd.Series)


def test_edge_calibration_uses_training_rows_only():
    score = np.arange(100, dtype=float)
    target = 2*score+1
    train = np.arange(100) < 80
    calibrated = _fit_edge_calibration(score, target, train)
    assert np.allclose(calibrated, target)
    changed = target.copy()
    changed[~train] = -999
    assert np.allclose(_fit_edge_calibration(score, changed, train), calibrated)
