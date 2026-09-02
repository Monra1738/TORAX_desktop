from __future__ import annotations

import numpy as np

from jaxa_torax.desktop.science_views import (
    energy_image,
    rgb_event_colors,
    rgb_from_channels,
    rgb_image,
    spectrum,
)
from jaxa_torax.desktop.viewer_3d_helpers import sync_scene_guides
from jaxa_torax.desktop.voxel_workflow import resolved_voxel_energy_band


class RefreshMixin:
    """Selective refresh paths: hidden products are never recomputed unnecessarily."""

    def _refresh_after_data_change(self, reset_camera=False):
        if self.state.region is None or not self.state.loaded_observations:
            self._refresh_dataset_summary()
            return
        if self.state.layout_mode != "2d":
            self._refresh_3d(reset_camera=reset_camera)
        if self.state.layout_mode != "3d":
            self._refresh_current_2d()
        self._refresh_spectrum()
        self._refresh_slice_products()
        self._refresh_dataset_summary()

    def _clear_loaded_observations(self):
        for key in list(self.state.observation_cache):
            self.workspace.three_d.remove_record(key)
        self.state.observation_cache.clear()
        self.state.loaded_observations.clear()
        self.state.visible_record_keys.clear()
        self.state.failed_observations.clear()
        self.state.pending_observation_keys.clear()
        self.state.combined_frame = self.state.combined_frame.iloc[0:0]
        self.state.clear_derived_caches()
        self._preview_products.clear()
        self.left_panel.set_observations([], set())
        self._invalidate_exact()

    def _refresh_dataset_summary(self):
        frame = self.state.displayed_frame()
        energy_range = self.state.energy_range()
        self.inspector.set_dataset(self.state.target_name, self.state.region, len(frame), energy_range)
        total_loaded = sum(obs.events_in_region for obs in self.state.loaded_observations)
        total_visible = sum(obs.events_in_region for obs in self.state.visible_observations())
        displayed = len(self.state.energy_filtered_frame())
        band = f"{self.state.energy_band[0]:.2f}–{self.state.energy_band[1]:.2f} keV"
        self.status_metrics.setText(
            f"{total_loaded:,} loaded region events  │  {total_visible:,} visible  │  "
            f"{len(frame):,} preview shown by default  │  {displayed:,} in selected band  │  {band}"
        )

    def _update_status(self, text: str):
        self.status_text.setText(text)

    def _effective_3d_energy_band(self):
        if self.state.spectrum_linked or self.state.filter_3d_by_energy:
            return self.state.energy_band
        return self.state.energy_range() or self.state.energy_band

    def _effective_3d_display_band(self):
        """Return the band used by every visible 3D representation."""
        if self.state.render_mode == "voxels":
            return resolved_voxel_energy_band(self.state)
        if getattr(self.state, "content_mode", "all") == "active":
            selected = self.state.selected_slice()
            if selected is not None:
                return float(selected.low_kev), float(selected.high_kev)
        return tuple(map(float, self.state.energy_band))

    def _spectrum_link_changed(self, linked: bool):
        linked = bool(linked)
        if self.state.spectrum_linked == linked:
            return
        self.state.spectrum_linked = linked
        self._invalidate_exact()
        self._schedule_viewer()
        self._schedule_analysis("2d")
        self._schedule_analysis("summary")
        self._workspace_changed()

    def _refresh_3d(self, reset_camera=False):
        if (
            self.state.layout_mode == "2d"
            or self.state.region is None
            or not self.state.loaded_observations
        ):
            return
        center_ra = self.state.region.center_ra_deg
        center_dec = self.state.region.center_dec_deg
        energy_band = self._effective_3d_display_band()
        self.workspace.three_d.set_scene_transform(
            center_ra,
            center_dec,
            self.state.region.radius_deg,
            energy_band,
            self.state.energy_display_scale,
        )
        density_strength = max(
            self.state.density_size_strength,
            self.state.density_opacity_strength,
        )
        if self.state.render_mode == "density":
            density_strength = max(0.7, density_strength)
        elif not (
            self.state.event_spatial_smoothing_arcmin > 0
            or self.state.event_energy_smoothing_kev > 0
        ):
            density_strength = 0.0

        if self.state.render_mode == "voxels":
            self.workspace.three_d.sync_voxel_actors(
                self.state.loaded_observations,
                self.state.visible_record_keys,
                center_ra,
                center_dec,
                energy_band,
                self.state.spatial_voxel_arcmin,
                self.state.energy_voxel_kev,
                self.state.voxel_spatial_smoothing_arcmin,
                self.state.voxel_energy_smoothing_kev,
                self.state.voxel_threshold_fraction,
                self.state.voxel_max_cells,
                self.state.voxel_opacity,
                self.state.voxel_show_edges,
                reset_camera=reset_camera,
                render=False,
            )
        else:
            self.workspace.three_d.sync_event_actors(
                self.state.loaded_observations,
                self.state.visible_record_keys,
                center_ra,
                center_dec,
                energy_band,
                self.state.event_color_mode,
                self.state.point_size,
                self.state.point_opacity,
                (
                    self.state.rgb_centers,
                    self.state.rgb_widths,
                    self.state.rgb_gains,
                    self.state.rgb_brightness,
                    self.state.rgb_gamma,
                ),
                density_strength,
                self.state.interactive_point_budget,
                reset_camera=reset_camera,
                render=False,
            )
            self.workspace.three_d.sync_slice_points(
                self.state.slices,
                self.state.loaded_observations,
                self.state.visible_record_keys,
                center_ra,
                center_dec,
                self.state.point_size,
                self.state.content_mode,
                self.state.selected_slice_uid,
                self.state.interactive_point_budget,
                render=False,
            )
            self.workspace.three_d.set_content_mode(
                self.state.content_mode,
                self.state.visible_record_keys,
                self.state.selected_slice_uid,
                render=False,
            )
        # One VTK render for observations + all slice planes.
        self.workspace.three_d.sync_slices(
            self.state.slices,
            self.state.region,
            center_dec,
            show_planes=self.state.show_slice_planes,
            render=False,
        )
        self.workspace.three_d.sync_reference_plane(
            self.state.region,
            self.state.energy_reference_kev,
            visible=self.state.show_energy_reference_plane,
            render=False,
        )
        sync_scene_guides(
            self.workspace.three_d, self.state.region, energy_band,
            self.state.show_grid_backdrop, self.state.show_slice_window,
            self.state.show_coordinate_triad, self.state.show_coordinate_values,
        )
        if self.workspace.three_d.available:
            self.workspace.three_d.plotter.render()

    def _refresh_current_2d(self):
        if self.state.layout_mode == "3d":
            return
        product = self.state.two_d_product
        if product == "rgb":
            self._refresh_rgb()
        elif product == "slice":
            self._refresh_selected_slice()
        elif product == "sky":
            self._refresh_sky()
        else:
            self._refresh_energy()

    def _cache_get(self, key):
        return self._preview_products.get(key)

    def _cache_put(self, key, value):
        self._preview_products[key] = value
        while len(self._preview_products) > 36:
            self._preview_products.pop(next(iter(self._preview_products)))
        return value

    def _band_count(self, frame, low: float, high: float) -> int:
        if frame.empty or "KEV" not in frame:
            return 0
        values = frame["KEV"].to_numpy(dtype=np.float32, copy=False)
        return int(np.count_nonzero((values >= float(low)) & (values <= float(high))))

    def _energy_product(self, frame, low: float, high: float, bins: int | None = None):
        bins = int(bins or self.state.image_bins)
        count = self._band_count(frame, low, high)
        smoothing = self.state.effective_image_smoothing(count)
        key = (
            "energy",
            self.state.data_signature(),
            round(float(low), 5),
            round(float(high), 5),
            bins,
            round(float(smoothing), 3),
        )
        product = self._cache_get(key)
        if product is None:
            product = energy_image(
                frame,
                low,
                high,
                bins=bins,
                center_ra=self.state.region.center_ra_deg,
                region=self.state.region,
                smoothing_sigma_pixels=smoothing,
            )
            self._cache_put(key, product)
        return product

    def _scalar_display_settings(self):
        return {
            "palette": self.state.image_palette,
            "stretch": self.state.image_stretch,
            "brightness": self.state.image_brightness,
            "contrast": self.state.image_contrast,
        }

    def _refresh_scalar_image_display(self):
        """Apply palette/LUT controls only; no data selection or image histogram work."""
        settings = self._scalar_display_settings()
        self.workspace.energy.refresh_scalar_display(**settings)
        self.workspace.slice.refresh_scalar_display(**settings)
        self.analysis.images.refresh_scalar_display(**settings)
        product = getattr(self, "_last_energy_product", None)
        if product is not None and self.state.region is not None:
            self.workspace.three_d.set_top_image(
                product,
                self.state.region,
                self.state.top_image_mode,
                self.state.top_image_opacity,
                **settings,
            )

    def _scalar_display_changed(self):
        """Apply a palette/LUT change immediately without scientific work."""
        self.state.image_palette = str(self.inspector.image_palette.currentData())
        self.state.image_stretch = str(self.inspector.image_stretch.currentData())
        self.state.image_brightness = self.inspector.image_brightness.value() / 100.0
        self.state.image_contrast = self.inspector.image_contrast.value() / 100.0
        self._refresh_scalar_image_display()
        self._workspace_changed()

    def _refresh_energy(self):
        if self.state.region is None:
            return
        if (
            self.state.energy_image_exact
            and self._exact_energy_data is not None
            and (
                (self.state.energy_image_exact_scope == "band" and self.state.filter_2d_by_energy)
                or (
                    self.state.energy_image_exact_scope == "all_events"
                    and not self.state.filter_2d_by_energy
                )
            )
        ):
            self._display_energy_product(self._exact_energy_product(self._exact_energy_data))
            return
        frame = self.state.displayed_frame()
        if self.state.spectrum_linked or self.state.filter_2d_by_energy:
            low, high = self.state.energy_band
        else:
            low, high = self.state.energy_range() or self.state.energy_band
        product = self._energy_product(frame, low, high)
        self._display_energy_product(product)

    def _refresh_rgb(self):
        if self.state.region is None:
            return
        if self.state.rgb_image_exact and self._exact_rgb_data is not None:
            data = self._exact_rgb_data
            rgb = rgb_from_channels(
                np.asarray(data["channels"], float), self.state.rgb_gains,
                self.state.rgb_brightness, self.state.rgb_gamma,
            )
            self._display_rgb_product(
                rgb, data["x_edges"], data["y_edges"], data["event_counts"]
            )
            return
        frame = self.state.displayed_frame()
        bands = self.state.rgb_bands()
        total = sum(self._band_count(frame, low, high) for low, high in bands)
        smoothing = self.state.effective_image_smoothing(max(1, total // 3))
        key = (
            "rgb",
            self.state.data_signature(),
            tuple((round(a, 4), round(b, 4)) for a, b in bands),
            int(self.state.image_bins),
            round(smoothing, 3),
            tuple(round(v, 3) for v in self.state.rgb_gains),
            round(self.state.rgb_brightness, 3),
            round(self.state.rgb_gamma, 3),
        )
        product = self._cache_get(key)
        if product is None:
            product = rgb_image(
                frame,
                bands,
                bins=self.state.image_bins,
                center_ra=self.state.region.center_ra_deg,
                region=self.state.region,
                smoothing_sigma_pixels=smoothing,
                gains=self.state.rgb_gains,
                brightness=self.state.rgb_brightness,
                gamma=self.state.rgb_gamma,
            )
            self._cache_put(key, product)
        rgb, x_edges, y_edges, counts = product
        self._display_rgb_product(rgb, x_edges, y_edges, counts)

    def _refresh_selected_slice(self):
        item = self.state.selected_slice()
        frame = self.state.displayed_frame()
        if item is None or self.state.region is None:
            return
        product = self._energy_product(frame, item.low_kev, item.high_kev)
        self.workspace.slice.set_scalar(
            product.values,
            product.x_edges,
            product.y_edges,
            f"{item.title}  •  {product.count:,} events",
            **self._scalar_display_settings(),
        )

    def _refresh_sky(self):
        # The event-map product is deliberately an all-energy view. The energy
        # image has its own independent band-filter checkbox.
        frame = self.state.displayed_frame()
        if self.state.region is None:
            self.workspace.sky.set_frame(frame, None)
            return
        rgb = None
        if self.state.event_color_mode == "rgb" and not frame.empty and "KEV" in frame:
            rgb = rgb_event_colors(
                frame["KEV"].to_numpy(dtype=float, copy=False),
                self.state.rgb_centers,
                self.state.rgb_widths,
                self.state.rgb_gains,
                self.state.rgb_brightness,
                self.state.rgb_gamma,
            )
        self.workspace.sky.set_frame(
            frame,
            self.state.region.center_ra_deg,
            self.state.event_color_mode,
            rgb=rgb,
        )

    def _display_energy_product(self, product):
        self._last_energy_product = product
        exact = self.state.energy_image_exact and (
            (self.state.energy_image_exact_scope == "band" and self.state.filter_2d_by_energy)
            or (
                self.state.energy_image_exact_scope == "all_events"
                and not self.state.filter_2d_by_energy
            )
        )
        label = "EXACT" if exact else "PREVIEW"
        exact_all = self.state.energy_image_exact_scope == "all_events" and self.state.energy_image_exact
        scope = "All energies" if exact_all else (
            "Selected band" if self.state.spectrum_linked or self.state.filter_2d_by_energy
            else "All energies"
        )
        title = (
            f"{scope}: {product.low_kev:.2f}–{product.high_kev:.2f} keV  •  "
            f"{product.count:,} events  •  {label}"
        )
        self.workspace.energy.set_scalar(
            product.values,
            product.x_edges,
            product.y_edges,
            title,
            **self._scalar_display_settings(),
        )
        self.inspector.energy_quality.setText(label)
        self.workspace.three_d.set_top_image(
            product,
            self.state.region,
            self.state.top_image_mode,
            self.state.top_image_opacity,
            **self._scalar_display_settings(),
        )

    def _display_rgb_product(self, rgb, x_edges, y_edges, counts):
        label = "EXACT" if self.state.rgb_image_exact else "PREVIEW"
        self.workspace.rgb.set_rgb(
            rgb,
            x_edges,
            y_edges,
            f"RGB composite  •  {counts}  •  {label}",
        )
        self.inspector.rgb_quality.setText(label)

    def _refresh_spectrum(self):
        frame = self.state.spectrum_frame()
        energy_range = self.state.energy_range() or (0.3, 12.0)
        bins = self.state.effective_spectrum_bins(len(frame))
        key = (
            "spectrum",
            self.state.visibility_signature(),
            None if self.state.spatial_rectangle is None else tuple(
                round(float(v), 6) for v in self.state.spatial_rectangle
            ),
            bins,
            round(float(energy_range[0]), 4),
            round(float(energy_range[1]), 4),
            round(float(self.state.spectrum_smoothing_bins), 3),
        )
        product = self._cache_get(key)
        if product is None:
            product = spectrum(
                frame,
                bins=bins,
                low=energy_range[0],
                high=energy_range[1],
                smoothing_sigma_bins=self.state.spectrum_smoothing_bins,
            )
            self._cache_put(key, product)
        comparison = None
        if self.state.comparison_rectangle is not None:
            comparison = spectrum(
                self.state.spectrum_frame_for(self.state.comparison_rectangle),
                bins=bins,
                low=energy_range[0],
                high=energy_range[1],
                smoothing_sigma_bins=self.state.spectrum_smoothing_bins,
            )
        self.analysis.spectrum.set_spectrum(
            product,
            self.state.spatial_rectangle,
            comparison,
            self.state.comparison_rectangle,
        )
        self.analysis.spectrum.set_effective_bins(bins, self.state.auto_spectrum_binning)
        self.analysis.spectrum.set_band(*self.state.energy_band)
        self.analysis.spectrum.set_slices(self.state.slices)

    def _refresh_slice_products(self, force_all: bool = False):
        visible = [item for item in self.state.slices if item.visible]
        frame = self.state.displayed_frame()
        if self.state.region is None or not visible:
            self.analysis.images.set_products([])
            self.analysis.profile.set_products([], [])
            return
        current = self.analysis.tabs.currentWidget()
        showing_all_slices = current in (self.analysis.images, self.analysis.profile)
        if not force_all and not showing_all_slices:
            if self.state.two_d_product == "slice":
                self._refresh_selected_slice()
            return
        products = [
            self._energy_product(frame, item.low_kev, item.high_kev, bins=140)
            for item in visible
        ]
        self.analysis.images.set_viewport(self.state.sky_viewport)
        self.analysis.images.set_products(
            products, [item.title for item in visible], self._scalar_display_settings()
        )
        self.analysis.profile.set_products(
            products, visible, self.state.selected_slice_uid
        )
        if self.state.selected_slice() is not None and self.state.two_d_product == "slice":
            self._refresh_selected_slice()

    def _refresh_analysis_products(self):
        pending = set(getattr(self, "_pending_refresh", set()))
        self._pending_refresh.clear()
        if not pending or "all" in pending:
            pending = {"spectrum", "2d", "slices", "summary"}
        if "spectrum" in pending:
            self._refresh_spectrum()
        if "2d" in pending:
            self._refresh_current_2d()
        if "slices" in pending:
            self._refresh_slice_products()
        if "summary" in pending:
            self._refresh_dataset_summary()

    def _schedule_viewer(self, delay_ms=None):
        if self.state.layout_mode != "2d":
            delay = self.state.viewer_debounce_ms if delay_ms is None else max(0, int(delay_ms))
            self._viewer_timer.start(delay)

    def _schedule_analysis(self, *kinds):
        if not kinds:
            kinds = ("all",)
        self._pending_refresh.update(kinds)
        self._analysis_timer.start(self.state.analysis_debounce_ms)
