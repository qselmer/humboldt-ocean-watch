from pathlib import Path


def test_streamlit_real_build_status_is_read_only_and_pilot_is_not_fallback() -> None:
    source = Path("app.py").read_text(encoding="utf-8")
    start = source.index("def load_real_climatology_build_status")
    end = source.index("@st.cache_resource", start)
    loader = source[start:end]
    assert "read_text" in loader
    assert "copernicus" not in loader.lower()
    assert "execute_build" not in loader
    assert "write" not in loader.replace("never preflight, build, download, or write", "")
    assert "never used as an operational fallback" in source
    assert "st.tabs" in source
