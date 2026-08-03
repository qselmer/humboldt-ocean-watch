from copy import deepcopy

from scripts.build_multidomain_sst_anomalies import build_multidomain_anomalies
from src.utils import load_config


def test_anomaly_builder_dry_run_writes_nothing(tmp_path) -> None:
    report = build_multidomain_anomalies(
        deepcopy(load_config()),
        allow_demo=True,
        dry_run=True,
        output_directory=tmp_path,
    )
    assert report["dry_run"] is True
    assert report["network_access"] is False
    assert not list(tmp_path.rglob("*"))


def test_streamlit_anomaly_section_is_read_only_and_eight_tabs() -> None:
    text = open("app.py", encoding="utf-8").read()
    assert "Multidomain climatology and anomaly status" in text
    assert "load_multidomain_anomaly_products" in text
    section = text.split("Multidomain climatology and anomaly status", 1)[1].split(
        'with st.expander("Quality control"', 1
    )[0]
    for forbidden in (
        "build_multidomain_anomalies(",
        "build_demo_climatology(",
        ".to_parquet(",
        ".to_netcdf(",
        "requests.",
    ):
        assert forbidden not in section
    assert text.count('"Data and methods"') >= 1
    assert "tabs = st.tabs(" in text
