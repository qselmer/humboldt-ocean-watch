"""Date-aware temporal feature engine for long-format metric series.

Magnitude definitions: positive/negative load are the mean positive/absolute
negative parts; accumulations are the discrete sums ``sum(max(x, 0))`` and
``sum(abs(min(x, 0)))``. Spectral variance is total variance recovered from
the one-sided periodogram. Mann-Kendall is approximated by scipy Kendall tau
between elapsed time and values, assuming independent observations.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.metric_result import MetricResult, not_calculated

TEMPORAL_COLUMNS = [
    "analysis_end_date", "window_start_date", "window_end_date", "window_days",
    "source_metric", "family", "metric", "value", "unit", "status", "reason",
    "n_observations", "valid_coverage",
]

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "state": ("mean", "median", "minimum", "maximum", "p10", "p25", "p75", "p90"),
    "magnitude": ("mean_absolute_value", "rms", "positive_load", "negative_load", "positive_accumulation", "negative_accumulation"),
    "variability": ("variance", "standard_deviation", "mad", "iqr", "p90_minus_p10", "coefficient_of_variation"),
    "instability": ("mean_absolute_successive_change", "rmssd", "maximum_absolute_jump", "sign_change_count", "turning_point_count", "proportion_changes_over_threshold"),
    "trend": ("ols_slope", "ols_intercept", "ols_r_squared", "theil_sen_slope", "theil_sen_ci_lower", "theil_sen_ci_upper", "mann_kendall_tau", "mann_kendall_p_value", "initial_final_difference"),
    "persistence": ("lag1_autocorrelation", "decorrelation_time", "longest_run_above_threshold", "mean_run_length_above_threshold", "longest_run_below_threshold", "proportion_above_threshold", "proportion_below_threshold", "state_transition_count"),
    "complexity": ("shannon_entropy", "spectral_variance", "dominant_period", "low_frequency_power_fraction", "high_frequency_power_fraction"),
}


def select_date_window(
    series: pd.DataFrame,
    analysis_end_date: str | pd.Timestamp,
    window: int | str | tuple[str | pd.Timestamp, str | pd.Timestamp],
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, int]:
    """Select by actual dates; integer windows include the ending calendar day."""
    if "date" not in series or "value" not in series:
        raise ValueError("Temporal input requires date and value columns")
    data = series.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date")
    end = pd.Timestamp(analysis_end_date)
    if isinstance(window, int):
        if window < 1:
            raise ValueError("Window days must be positive")
        start = end - pd.Timedelta(days=window - 1)
        days = window
    elif isinstance(window, tuple):
        start, requested_end = map(pd.Timestamp, window)
        if requested_end < start:
            raise ValueError("User-selected period ends before it starts")
        end = min(end, requested_end)
        days = int((end.normalize() - start.normalize()).days + 1)
    elif window == "all":
        eligible = data.loc[data.date <= end, "date"]
        start = eligible.min() if not eligible.empty else end
        days = max(1, int((end.normalize() - start.normalize()).days + 1))
    else:
        raise ValueError("Window must be days, 'all', or a (start, end) period")
    selected = data[(data.date >= start) & (data.date <= end)].copy()
    return selected, start, end, days


def _result(name: str, family: str, value: float, n: int, coverage: float,
            unit: str, days: int, reason: str | None = None) -> MetricResult:
    return MetricResult(
        metric=name, value=value, status="warning" if reason else "valid", reason=reason,
        n_observations=n, valid_coverage=coverage, unit=unit, family=family,
        window_days=days,
    )


def _missing(name: str, family: str, reason: str, n: int, coverage: float,
             unit: str, days: int) -> MetricResult:
    return not_calculated(
        name, reason, unit=unit, family=family, n_observations=n,
        valid_coverage=coverage, window_days=days,
    )


def _feature_unit(family: str, name: str, source_unit: str) -> str:
    """Return stable units for calculated and unavailable feature results."""
    if family == "variability":
        return "1" if name == "coefficient_of_variation" else f"{source_unit} squared" if name == "variance" else source_unit
    if family == "instability":
        return "count" if name.endswith("count") else "1" if name.startswith("proportion") else source_unit
    if family == "trend":
        if name in {"ols_r_squared", "mann_kendall_tau", "mann_kendall_p_value"}:
            return "1"
        return f"{source_unit} per day" if name.startswith("ols_slope") or name.startswith("theil_sen") else source_unit
    if family == "persistence":
        if "run" in name or name == "decorrelation_time":
            return "days"
        return "count" if name == "state_transition_count" else "1"
    if family == "complexity":
        if name == "shannon_entropy":
            return "bits"
        if name == "spectral_variance":
            return f"{source_unit} squared"
        return "days" if name == "dominant_period" else "1"
    return source_unit


def _runs(state: np.ndarray, dates: pd.DatetimeIndex) -> list[int]:
    """Return true-state run lengths without joining across calendar gaps."""
    runs: list[int] = []
    current = 0
    for index, active in enumerate(state):
        gap = index > 0 and (dates[index] - dates[index - 1]).days != 1
        if gap or not active:
            if current:
                runs.append(current)
            current = 0
        if active:
            current += 1
    if current:
        runs.append(current)
    return runs


def compute_temporal_features(
    series: pd.DataFrame,
    *,
    source_metric: str,
    analysis_end_date: str | pd.Timestamp,
    window: int | str | tuple[str | pd.Timestamp, str | pd.Timestamp],
    minimum_observations: int = 5,
    instability_threshold: float = 0.5,
    entropy_bins: int = 10,
    spectral_minimum_observations: int = 30,
    persistence_threshold: float = 0.0,
    unit: str = "1",
    coefficient_of_variation_valid: bool | None = None,
) -> tuple[list[MetricResult], pd.Timestamp, pd.Timestamp]:
    """Calculate all temporal families for one source metric and end date."""
    selected, start, end, days = select_date_window(series, analysis_end_date, window)
    values_all = pd.to_numeric(selected.value, errors="coerce").to_numpy(dtype=float)
    dates_all = pd.DatetimeIndex(selected.date)
    finite = np.isfinite(values_all)
    values = values_all[finite]
    dates = dates_all[finite]
    n = len(values)
    coverage = n / days
    results: list[MetricResult] = []
    all_names = [(family, name) for family, names in FEATURE_FAMILIES.items() for name in names]
    if n < minimum_observations:
        reason = f"Only {n} finite observations; minimum is {minimum_observations}"
        return ([_missing(name, family, reason, n, coverage, _feature_unit(family, name, unit), days) for family, name in all_names], start, end)

    q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
    mean, median = float(np.mean(values)), float(np.median(values))
    state = {
        "mean": mean, "median": median, "minimum": float(np.min(values)), "maximum": float(np.max(values)),
        "p10": float(q10), "p25": float(q25), "p75": float(q75), "p90": float(q90),
    }
    positive = np.maximum(values, 0.0)
    negative = np.abs(np.minimum(values, 0.0))
    magnitude = {
        "mean_absolute_value": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values**2))),
        "positive_load": float(np.mean(positive)), "negative_load": float(np.mean(negative)),
        "positive_accumulation": float(np.sum(positive)), "negative_accumulation": float(np.sum(negative)),
    }
    variability = {
        "variance": float(np.var(values)), "standard_deviation": float(np.std(values)),
        "mad": float(np.median(np.abs(values - median))), "iqr": float(q75 - q25),
        "p90_minus_p10": float(q90 - q10),
    }
    # CV is not invariant to an arbitrary zero point, so Celsius temperature,
    # anomalies, and standardized indices are not enabled automatically. A
    # caller may opt in only for a scientifically justified ratio-scale input.
    cv_allowed = bool(coefficient_of_variation_valid)
    variability["coefficient_of_variation"] = float(np.std(values) / abs(mean)) if cv_allowed and not np.isclose(mean, 0.0) else np.nan
    for family, mapping in (("state", state), ("magnitude", magnitude)):
        results.extend(_result(name, family, value, n, coverage, unit, days) for name, value in mapping.items())
    for name, value in variability.items():
        if name == "coefficient_of_variation" and not np.isfinite(value):
            results.append(_missing(name, "variability", "Coefficient of variation is invalid for this scale or a zero mean", n, coverage, "1", days))
        else:
            value_unit = "1" if name == "coefficient_of_variation" else f"{unit} squared" if name == "variance" else unit
            results.append(_result(name, "variability", value, n, coverage, value_unit, days))

    consecutive = np.diff(dates.values).astype("timedelta64[D]").astype(int) == 1
    raw_differences = np.diff(values)
    differences = raw_differences[consecutive]
    if differences.size:
        signs = np.sign(values)
        sign_pairs = consecutive & (signs[:-1] != 0) & (signs[1:] != 0)
        sign_changes = int(np.sum(sign_pairs & (signs[:-1] != signs[1:])))
        turning = int(np.sum(consecutive[:-1] & consecutive[1:] & (raw_differences[:-1] * raw_differences[1:] < 0))) if len(raw_differences) > 1 else 0
        instability = {
            "mean_absolute_successive_change": float(np.mean(np.abs(differences))),
            "rmssd": float(np.sqrt(np.mean(differences**2))),
            "maximum_absolute_jump": float(np.max(np.abs(differences))),
            "sign_change_count": float(sign_changes), "turning_point_count": float(turning),
            "proportion_changes_over_threshold": float(np.mean(np.abs(differences) > instability_threshold)),
        }
        results.extend(_result(name, "instability", value, len(differences), len(differences) / max(1, days - 1), "count" if name.endswith("count") else "1" if name.startswith("proportion") else unit, days) for name, value in instability.items())
    else:
        results.extend(_missing(name, "instability", "No consecutive finite daily pairs; temporal gaps are not bridged", n, coverage, _feature_unit("instability", name, unit), days) for name in FEATURE_FAMILIES["instability"])

    elapsed = (dates - dates[0]).total_seconds().to_numpy() / 86400
    if n >= max(3, minimum_observations) and np.unique(elapsed).size >= 2:
        ols = stats.linregress(elapsed, values)
        theil = stats.theilslopes(values, elapsed, alpha=0.95)
        kendall = stats.kendalltau(elapsed, values, nan_policy="omit")
        trend = {
            "ols_slope": float(ols.slope), "ols_intercept": float(ols.intercept),
            "ols_r_squared": float(ols.rvalue**2), "theil_sen_slope": float(theil.slope),
            "theil_sen_ci_lower": float(theil.low_slope), "theil_sen_ci_upper": float(theil.high_slope),
            "mann_kendall_tau": float(kendall.statistic), "mann_kendall_p_value": float(kendall.pvalue),
            "initial_final_difference": float(values[-1] - values[0]),
        }
        for name, value in trend.items():
            trend_unit = f"{unit} per day" if name.startswith("ols_slope") or name.startswith("theil_sen") else unit
            if name in {"ols_r_squared", "mann_kendall_tau", "mann_kendall_p_value"}:
                trend_unit = "1"
            results.append(_result(name, "trend", value, n, coverage, trend_unit, days))
    else:
        results.extend(_missing(name, "trend", "Insufficient distinct temporal observations for trend", n, coverage, _feature_unit("trend", name, unit), days) for name in FEATURE_FAMILIES["trend"])

    above = values > persistence_threshold
    below = values < persistence_threshold
    transitions = int(np.sum(consecutive & (above[1:] != above[:-1])))
    above_runs, below_runs = _runs(above, dates), _runs(below, dates)
    persistence: dict[str, float] = {
        "longest_run_above_threshold": float(max(above_runs, default=0)),
        "mean_run_length_above_threshold": float(np.mean(above_runs)) if above_runs else 0.0,
        "longest_run_below_threshold": float(max(below_runs, default=0)),
        "proportion_above_threshold": float(np.mean(above)),
        "proportion_below_threshold": float(np.mean(below)),
        "state_transition_count": float(transitions),
    }
    lagged_x = values[:-1][consecutive]
    lagged_y = values[1:][consecutive]
    if lagged_x.size >= 2 and np.std(lagged_x) > 0 and np.std(lagged_y) > 0:
        autocorrelation = float(np.corrcoef(lagged_x, lagged_y)[0, 1])
        persistence["lag1_autocorrelation"] = autocorrelation
        persistence["decorrelation_time"] = float(-1.0 / np.log(autocorrelation)) if 0 < autocorrelation < 1 else np.nan
    else:
        persistence["lag1_autocorrelation"] = np.nan
        persistence["decorrelation_time"] = np.nan
    for name in FEATURE_FAMILIES["persistence"]:
        value = persistence[name]
        if not np.isfinite(value):
            reason = "Lag-1 autocorrelation is invalid" if name == "lag1_autocorrelation" else "Decorrelation time requires lag-1 autocorrelation strictly between 0 and 1"
            results.append(_missing(name, "persistence", reason, n, coverage, _feature_unit("persistence", name, unit), days))
        else:
            results.append(_result(name, "persistence", value, n, coverage, _feature_unit("persistence", name, unit), days))

    counts, _ = np.histogram(values, bins=entropy_bins)
    probabilities = counts[counts > 0] / counts.sum()
    results.append(_result("shannon_entropy", "complexity", float(-np.sum(probabilities * np.log2(probabilities))), n, coverage, "bits", days))
    regular = n == days and bool(np.all(np.diff(dates.values).astype("timedelta64[D]").astype(int) == 1))
    if regular and n >= spectral_minimum_observations and np.std(values) > 0:
        centered = values - mean
        power = np.abs(np.fft.rfft(centered)) ** 2
        frequencies = np.fft.rfftfreq(n, d=1.0)
        multipliers = np.full(power.size, 2.0)
        multipliers[0] = 1.0
        if n % 2 == 0:
            multipliers[-1] = 1.0
        spectral_variance = float(np.sum(power * multipliers) / n**2)
        power, frequencies = power[1:], frequencies[1:]
        multipliers = multipliers[1:]
        power = power * multipliers
        total_power = power.sum()
        dominant = float(1.0 / frequencies[np.argmax(power)])
        split = 0.1
        spectral = {
            "spectral_variance": spectral_variance, "dominant_period": dominant,
            "low_frequency_power_fraction": float(power[frequencies <= split].sum() / total_power),
            "high_frequency_power_fraction": float(power[frequencies > split].sum() / total_power),
        }
        results.extend(_result(
            name, "complexity", value, n, coverage,
            "days" if name == "dominant_period" else f"{unit} squared" if name == "spectral_variance" else "1",
            days,
        ) for name, value in spectral.items())
    else:
        reason = "Spectral features require regular, finite, non-constant sampling with sufficient observations"
        results.extend(_missing(name, "complexity", reason, n, coverage, _feature_unit("complexity", name, unit), days) for name in FEATURE_FAMILIES["complexity"] if name != "shannon_entropy")
    return results, start, end


def build_temporal_feature_table(
    bank: pd.DataFrame,
    *,
    source_metrics: Iterable[str],
    windows: Iterable[int | str | tuple[str | pd.Timestamp, str | pd.Timestamp]],
    minimum_observations: int,
    instability_threshold: float,
    entropy_bins: int,
    spectral_minimum_observations: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_metric in source_metrics:
        source = bank[bank.metric == source_metric].sort_values("date")
        if source.empty:
            continue
        unit = str(source.unit.dropna().iloc[0]) if source.unit.notna().any() else "1"
        for end in pd.DatetimeIndex(source.date.unique()):
            for window in windows:
                results, start, actual_end = compute_temporal_features(
                    source[["date", "value"]], source_metric=source_metric,
                    analysis_end_date=end, window=window,
                    minimum_observations=minimum_observations,
                    instability_threshold=instability_threshold,
                    entropy_bins=entropy_bins,
                    spectral_minimum_observations=spectral_minimum_observations,
                    unit=unit,
                )
                for result in results:
                    rows.append({
                        "analysis_end_date": end, "window_start_date": start,
                        "window_end_date": actual_end, "window_days": result.window_days,
                        "source_metric": source_metric, "family": result.family,
                        "metric": result.metric, "value": result.value, "unit": result.unit,
                        "status": result.status, "reason": result.reason,
                        "n_observations": result.n_observations,
                        "valid_coverage": result.valid_coverage,
                    })
    return pd.DataFrame(rows, columns=TEMPORAL_COLUMNS)
