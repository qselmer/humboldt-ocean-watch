"""Reusable offline overlays for the Increment 6A geographic foundation."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from shapely.geometry.base import BaseGeometry

from src.coastline_source import CoastlineSourceStatus, LocalGeometries
from src.geography import (
    GeographicBounds,
    GeographicDomain,
    GeographyRegistry,
    StandardRegion,
    geometry_is_valid,
    rectangle_vertices,
)

try:
    import cartopy.crs as ccrs
except ImportError:  # pragma: no cover - optional fallback.
    ccrs = None


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeographicOverlayStyle:
    """Centralized non-scientific styling for geographic outlines."""

    display_color: str = "#404040"
    coastal_domain_color: str = "#0072B2"
    pressure_domain_color: str = "#7B3294"
    standard_region_color: str = "#D55E00"
    corridor_color: str = "#009E73"
    land_facecolor: str = "#F2F2F2"
    coastline_color: str = "#111111"
    border_color: str = "#555555"
    ocean_facecolor: str = "#EAF4FA"
    label_color: str = "#111111"


DEFAULT_STYLE = GeographicOverlayStyle()


def configure_pacific_context_extent(
    axis: object,
    bounds: GeographicBounds,
    *,
    use_cartopy: bool,
) -> None:
    extent = [bounds.west, bounds.east, bounds.south, bounds.north]
    if use_cartopy:
        axis.set_extent(extent, crs=ccrs.PlateCarree())
        grid = axis.gridlines(
            draw_labels=True, color="0.68", alpha=0.65, linewidth=0.55, linestyle=":"
        )
        grid.top_labels = False
        grid.right_labels = False
        grid.xlabel_style = {"color": "black", "size": 8}
        grid.ylabel_style = {"color": "black", "size": 8}
    else:
        axis.set_xlim(bounds.west, bounds.east)
        axis.set_ylim(bounds.south, bounds.north)
        axis.set_xlabel("Longitude (degrees east)")
        axis.set_ylabel("Latitude (degrees north)")
        axis.grid(color="0.75", alpha=0.7, linewidth=0.55, linestyle=":")


def _rectangle_patch(
    bounds: GeographicBounds,
    *,
    edgecolor: str,
    linewidth: float,
    linestyle: str,
    zorder: int,
    use_cartopy: bool,
) -> Rectangle:
    options: dict[str, object] = {}
    if use_cartopy:
        options["transform"] = ccrs.PlateCarree()
    return Rectangle(
        (bounds.west, bounds.south),
        bounds.width,
        bounds.height,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
        **options,
    )


def add_domain_outline(
    axis: object,
    domain: GeographicDomain,
    *,
    edgecolor: str,
    linewidth: float = 1.4,
    linestyle: str = "--",
    use_cartopy: bool,
    label: bool = True,
    label_longitude: float | None = None,
    label_latitude: float | None = None,
    label_rotation: float = 0.0,
    zorder: int = 8,
) -> None:
    axis.add_patch(
        _rectangle_patch(
            domain.bounds,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
            use_cartopy=use_cartopy,
        )
    )
    if label:
        text_options = {"transform": ccrs.PlateCarree()} if use_cartopy else {}
        axis.text(
            domain.bounds.west + 0.8 if label_longitude is None else label_longitude,
            domain.bounds.north - 1.4 if label_latitude is None else label_latitude,
            domain.label,
            color=edgecolor,
            fontsize=8,
            weight="bold",
            va="top",
            rotation=label_rotation,
            zorder=zorder + 1,
            **text_options,
        )


def add_standard_region_boxes(
    axis: object,
    regions: list[StandardRegion],
    *,
    style: GeographicOverlayStyle = DEFAULT_STYLE,
    use_cartopy: bool,
    show_labels: bool = True,
) -> None:
    text_options = {"transform": ccrs.PlateCarree()} if use_cartopy else {}
    for region in regions:
        axis.add_patch(
            _rectangle_patch(
                region.bounds,
                edgecolor=style.standard_region_color,
                linewidth=1.7,
                linestyle="-",
                zorder=12,
                use_cartopy=use_cartopy,
            )
        )
        if show_labels:
            axis.text(
                region.central_longitude,
                region.bounds.north + 0.7,
                region.label,
                ha="center",
                va="bottom",
                fontsize=8,
                color=style.standard_region_color,
                weight="bold",
                zorder=13,
                **text_options,
            )


def _iter_polygon_exteriors(geometry: BaseGeometry):
    if geometry.geom_type == "Polygon":
        yield geometry.exterior
    elif geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            yield polygon.exterior
    else:
        for part in getattr(geometry, "geoms", ()):
            yield from _iter_polygon_exteriors(part)


def _iter_lines(geometry: BaseGeometry):
    if geometry.geom_type in {"LineString", "LinearRing"}:
        yield geometry
    else:
        for part in getattr(geometry, "geoms", ()):
            yield from _iter_lines(part)


def _add_geometry(
    axis: object,
    geometry: BaseGeometry,
    *,
    use_cartopy: bool,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    if use_cartopy:
        axis.add_geometries(
            [geometry],
            crs=ccrs.PlateCarree(),
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
        return
    if geometry.geom_type in {"Polygon", "MultiPolygon", "GeometryCollection"}:
        for exterior in _iter_polygon_exteriors(geometry):
            x, y = exterior.xy
            axis.fill(
                x,
                y,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )
    else:
        for line in _iter_lines(geometry):
            x, y = line.xy
            axis.plot(
                x,
                y,
                color=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )


def add_coastal_corridor(
    axis: object,
    corridor: BaseGeometry,
    *,
    style: GeographicOverlayStyle = DEFAULT_STYLE,
    use_cartopy: bool,
) -> None:
    if not geometry_is_valid(corridor):
        raise ValueError("Coastal corridor geometry must be finite and valid")
    _add_geometry(
        axis,
        corridor,
        use_cartopy=use_cartopy,
        facecolor=style.corridor_color,
        edgecolor=style.corridor_color,
        linewidth=0.7,
        alpha=0.30,
        zorder=7,
    )


def _render_foundation(
    registry: GeographyRegistry,
    *,
    local_geometries: LocalGeometries | None,
    corridor: BaseGeometry | None,
    source_status: CoastlineSourceStatus,
    style: GeographicOverlayStyle,
    use_cartopy: bool,
) -> matplotlib.figure.Figure:
    subplot = {"projection": ccrs.PlateCarree()} if use_cartopy else {}
    figure, axis = plt.subplots(
        figsize=(13.0, 7.2),
        constrained_layout=False,
        facecolor="white",
        subplot_kw=subplot,
    )
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.13, top=0.90)
    axis.set_facecolor(style.ocean_facecolor)
    configure_pacific_context_extent(
        axis, registry.display_domain.bounds, use_cartopy=use_cartopy
    )

    if local_geometries is not None and local_geometries.all_land is not None:
        _add_geometry(
            axis,
            local_geometries.all_land,
            use_cartopy=use_cartopy,
            facecolor=style.land_facecolor,
            edgecolor=style.coastline_color,
            linewidth=0.45,
            alpha=1.0,
            zorder=3,
        )
    if local_geometries is not None:
        _add_geometry(
            axis,
            local_geometries.display_coastline,
            use_cartopy=use_cartopy,
            facecolor="none",
            edgecolor=style.coastline_color,
            linewidth=0.85,
            alpha=1.0,
            zorder=5,
        )
        if local_geometries.borders is not None:
            _add_geometry(
                axis,
                local_geometries.borders,
                use_cartopy=use_cartopy,
                facecolor="none",
                edgecolor=style.border_color,
                linewidth=0.45,
                alpha=0.9,
                zorder=5,
            )

    if corridor is not None and registry.map_overlays.get("show_coastal_corridor", True):
        add_coastal_corridor(axis, corridor, style=style, use_cartopy=use_cartopy)

    if registry.map_overlays.get("show_analysis_domains", True):
        for identifier, domain in registry.analysis_domains.items():
            if identifier == "pacific_context":
                color, linestyle, width = style.display_color, "--", 1.2
            elif identifier == "humboldt_coastal":
                color, linestyle, width = style.coastal_domain_color, "--", 1.7
            else:
                color, linestyle, width = style.pressure_domain_color, "-.", 1.7
            add_domain_outline(
                axis,
                domain,
                edgecolor=color,
                linewidth=width,
                linestyle=linestyle,
                use_cartopy=use_cartopy,
                label=registry.map_overlays.get("show_labels", True),
                label_longitude=(
                    domain.bounds.west + 1.0
                    if identifier == "humboldt_coastal"
                    else None
                ),
                label_latitude=(
                    domain.bounds.central_latitude
                    if identifier == "humboldt_coastal"
                    else None
                ),
                label_rotation=(90.0 if identifier == "humboldt_coastal" else 0.0),
            )

    if registry.map_overlays.get("show_standard_regions", True):
        add_standard_region_boxes(
            axis,
            list(registry.standard_regions.values()),
            style=style,
            use_cartopy=use_cartopy,
            show_labels=registry.map_overlays.get("show_labels", True),
        )

    text_options = {"transform": ccrs.PlateCarree()} if use_cartopy else {}
    axis.text(
        -137.0,
        -25.0,
        "Pacific Ocean",
        color="#2B6F8F",
        fontsize=12,
        style="italic",
        ha="center",
        zorder=2,
        **text_options,
    )
    if local_geometries is not None and registry.map_overlays.get("show_labels", True):
        for name, longitude, latitude in (
            ("Ecuador", -78.4, -1.2),
            ("Peru", -75.3, -10.0),
            ("Chile", -72.5, -30.0),
        ):
            axis.text(
                longitude,
                latitude,
                name,
                fontsize=8,
                color=style.label_color,
                ha="center",
                weight="bold",
                zorder=9,
                **text_options,
            )

    legend = [
        Line2D([0], [0], color=style.standard_region_color, linewidth=1.7, label="Standard Niño region"),
        Line2D([0], [0], color=style.coastal_domain_color, linewidth=1.7, linestyle="--", label="Humboldt coastal analysis domain"),
        Line2D([0], [0], color=style.pressure_domain_color, linewidth=1.7, linestyle="-.", label="South Pacific High diagnostic domain"),
    ]
    if corridor is not None:
        legend.append(
            Patch(
                facecolor=style.corridor_color,
                edgecolor=style.corridor_color,
                alpha=0.30,
                label="Mainland-derived Humboldt corridor (60 nm)",
            )
        )
    axis.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.95)
    figure.suptitle(
        "Humboldt Ocean Watch geographic foundation", fontsize=15, y=0.965
    )
    figure.text(
        0.5,
        0.035,
        "Outlines are display and diagnostic domains, not official jurisdictional boundaries. "
        "Current scientific calculations remain restricted to Niño 1+2.",
        ha="center",
        fontsize=8,
        color="0.25",
    )
    if corridor is None:
        axis.text(
            0.5,
            0.035,
            source_status.message,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#9C2F21",
            bbox={"facecolor": "white", "edgecolor": "#9C2F21", "alpha": 0.9},
            zorder=20,
        )
    return figure


def plot_geographic_foundation(
    registry: GeographyRegistry,
    *,
    local_geometries: LocalGeometries | None,
    corridor: BaseGeometry | None,
    source_status: CoastlineSourceStatus,
    style: GeographicOverlayStyle = DEFAULT_STYLE,
) -> matplotlib.figure.Figure:
    """Plot future geographic scope without displaying scientific fields."""
    if ccrs is not None:
        figure: matplotlib.figure.Figure | None = None
        try:
            figure = _render_foundation(
                registry,
                local_geometries=local_geometries,
                corridor=corridor,
                source_status=source_status,
                style=style,
                use_cartopy=True,
            )
            figure.canvas.draw()
            return figure
        except Exception as exc:
            LOGGER.warning("Cartopy foundation rendering failed; using fallback: %s", exc)
            if figure is not None:
                plt.close(figure)
    return _render_foundation(
        registry,
        local_geometries=local_geometries,
        corridor=corridor,
        source_status=source_status,
        style=style,
        use_cartopy=False,
    )
