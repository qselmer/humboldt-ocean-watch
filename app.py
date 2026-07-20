"""Minimal Streamlit interface for Humboldt Ocean Watch."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.data_loader import load_sst_dataset
from src.utils import configure_logging, load_config, resolve_project_path

st.set_page_config(page_title="Humboldt Ocean Watch", page_icon="🌊", layout="wide")
config = load_config()
configure_logging(os.getenv("HOW_LOG_LEVEL", config["logging"]["level"]))

st.title("Humboldt Ocean Watch")
st.warning("Experimental thermal monitoring product for the Niño 1+2 region.")
st.caption(
    "This application provides descriptive SST monitoring only and does not classify "
    "the official magnitude of El Niño Costero."
)

default_path = os.getenv("HOW_SST_PATH", config["data"]["input_path"])
requested_path = st.text_input("Local SST NetCDF path", value=default_path)

try:
    source = resolve_project_path(Path(requested_path))
    region = config["region"]
    dataset = load_sst_dataset(
        source,
        aliases=config["data"]["sst_aliases"],
        demo_options={
            "longitude": tuple(region["longitude"]),
            "latitude": tuple(region["latitude"]),
            "days": config["data"]["demo_days"],
            "seed": config["data"]["demo_seed"],
        },
    )
    st.success(f"Dataset ready: {source}")
    first_date = str(dataset.time.min().dt.strftime("%Y-%m-%d").item())
    last_date = str(dataset.time.max().dt.strftime("%Y-%m-%d").item())
    columns = st.columns(4)
    columns[0].metric("Start date", first_date)
    columns[1].metric("End date", last_date)
    columns[2].metric("Dimensions", " × ".join(f"{key}: {value}" for key, value in dataset.sizes.items()))
    columns[3].metric("Temperature units", dataset["sst"].attrs["units"])
    st.write("Study region", {"longitude": region["longitude"], "latitude": region["latitude"]})
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(f"Unable to prepare SST dataset: {exc}")
    st.stop()
