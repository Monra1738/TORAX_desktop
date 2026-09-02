from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from jaxa_udon3.desktop.science_views import (
    EnergySceneTransform,
    SkyViewport,
    energy_gradient_colors,
    mission_event_colors,
    point_density,
    rgb_event_colors,
    voxel_histogram,
    wrapped_ra,
)
from jaxa_udon3.desktop.theme import PLOT_TEXT, VIEWPORT
from jaxa_udon3.desktop.viewer_3d_helpers import (
    balanced_scene_transform,
    exclude_actor_from_bounds,
    reference_plane,
    set_voxel_actor_appearance,
    transformed_voxel_geometry,
)

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception:
    pv = None
    QtInteractor = None
def record_key(obs) -> str:
    return f"{obs.record.mission}/{obs.record.instrument}/{obs.record.observation_id}"
def _sample_frame(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    index = np.linspace(0, len(frame) - 1, max(1, int(limit)), dtype=np.int64)
    return frame.iloc[index]


def enabled_slice_point_uids(slices, content_mode="all", active_uid=None) -> set[str]:
    """Select colored slice-point overlays for one unambiguous content mode."""
    if content_mode in ("active", "all_active"):
        return {
            item.uid
            for item in slices
            if item.uid == active_uid and item.show_points
        }
    if content_mode == "multiple":
        return {item.uid for item in slices if item.visible and item.show_points}
    return set()


def _edges_for_grid(show_edges: bool, product) -> bool:
    """Avoid expensive per-cell VTK edges for very large voxel grids."""
    if not show_edges or product is None:
        return False
    hist, _edges = product
    return hist.size <= 100_000
class ThreeDView(QWidget):
    event_selected = Signal(object)
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.available = pv is not None and QtInteractor is not None and not os.getenv("UDON3_DISABLE_3D")
        self._plotter = None
        self._event_actors, self._density_actors = {}, {}
        self._voxel_actors, self._spectral_actors = {}, {}
        self._slice_actors, self._slice_point_actors = {}, {}
        self._slice_point_signatures: dict[str, tuple] = {}
        self._top_image_actor, self._reference_actor = None, None
        self._backdrop_actor = None
        self._event_signatures, self._density_signatures, self._voxel_signatures = {}, {}, {}
        self._voxel_products, self._voxel_product_signatures = {}, {}
        self._slice_signatures, self._pick_frames, self._pick_points = {}, {}, {}
        self._axes_ready = False
        self._scene_transform = None
        self._energy_axis_band = None
        if self.available:
            self._plotter = QtInteractor(self)
            self._plotter.set_background(VIEWPORT, top="#0b1628")
            layout.addWidget(self._plotter.interactor)
            try:
                self._plotter.enable_point_picking(
                    callback=self._picked,
                    show_message=False,
                    show_point=False,
                )
            except Exception:
                pass
        else:
            label = QLabel("3D viewer unavailable\nInstall PyVista + PyVistaQt for native VTK rendering.")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size:14px;color:#94a3b8;background:#030712;")
            layout.addWidget(label)
    @property
    def plotter(self):
        return self._plotter
    def camera_state(self):
        if not self.available or self._plotter is None:
            return None
        try:
            return [list(map(float, point)) for point in self._plotter.camera_position]
        except Exception:
            return None
    def restore_camera_state(self, state):
        if not state or not self.available or self._plotter is None:
            return
        try:
            if len(state) == 3 and all(len(point) == 3 for point in state):
                self._plotter.camera_position = tuple(tuple(map(float, point)) for point in state)
                self._plotter.render()
        except Exception:
            pass
    @staticmethod
    def _coordinates(frame, center_ra, center_dec):
        ra = wrapped_ra(frame["RA"].to_numpy(dtype=np.float32, copy=False), center_ra)
        dec = frame["DEC"].to_numpy(dtype=np.float32, copy=False)
        energy = frame["KEV"].to_numpy(dtype=np.float32, copy=False)
        return np.column_stack((ra, dec, energy)).astype(np.float32, copy=False)
    def set_scene_transform(
        self, center_ra, center_dec, radius_deg, energy_band, depth_aspect=1.0
    ):
        transform = balanced_scene_transform(
            center_ra, center_dec, radius_deg, energy_band, depth_aspect
        )
        changed = transform != self._scene_transform
        self._scene_transform = transform
        if changed:
            self._axes_ready = False
            self._energy_axis_band = None
            # Actor geometry is stored in scene coordinates.  Invalidate only
            # those lightweight display signatures when the active band or
            # depth aspect changes; cached voxel histograms remain reusable.
            self._event_signatures.clear()
            self._density_signatures.clear()
            self._voxel_signatures.clear()
            self._slice_signatures.clear()
            self._slice_point_signatures.clear()
    def _display_coordinates(self, frame, center_ra, center_dec):
        transform = self._scene_transform
        return self._coordinates(frame, center_ra, center_dec) if transform is None or transform.center_ra_deg != float(center_ra) else transform.coordinates(frame)
    def _ensure_axes(self):
        if not self.available or self._axes_ready:
            return
        try:
            self._plotter.show_bounds(
                grid="front",
                location="outer",
                color=PLOT_TEXT,
                xtitle="Right Ascension (deg)",
                ytitle="Declination (deg)",
                ztitle="Energy (keV)",
                font_size=10,
                all_edges=True,
            )
        except Exception:
            self._plotter.add_axes(
                xlabel="RA (+ left)",
                ylabel="DEC (deg)",
                zlabel="Energy (keV)",
                color=PLOT_TEXT,
            )
        self._axes_ready = True
    def _update_point_actor(self, actor, cloud, point_size: float, opacity: float) -> bool:
        try:
            actor.mapper.dataset = cloud
            try:
                actor.prop.point_size = float(point_size)
                actor.prop.opacity = float(opacity)
            except Exception:
                actor.GetProperty().SetPointSize(float(point_size))
                actor.GetProperty().SetOpacity(float(opacity))
            return True
        except Exception:
            return False
    def _set_point_actor_appearance(self, actor, point_size: float, opacity: float):
        try:
            actor.prop.point_size = float(point_size)
            actor.prop.opacity = float(opacity)
        except Exception:
            try:
                actor.GetProperty().SetPointSize(float(point_size))
                actor.GetProperty().SetOpacity(float(opacity))
            except Exception:
                pass
    @staticmethod
    def display_budgets(observations, visible_keys: set[str], total_budget: int) -> dict[str, int]:
        visible = [item for item in observations if record_key(item) in visible_keys]
        if not visible:
            return {}
        total = max(len(visible), int(total_budget))
        base = min(750, total // len(visible))
        remaining = total - base * len(visible)
        weights = np.sqrt(np.asarray([max(1, len(item.frame)) for item in visible], dtype=float))
        weights /= weights.sum() or 1.0
        extras = np.floor(weights * remaining).astype(int)
        for index in np.argsort(-(weights * remaining - extras))[:remaining - int(extras.sum())]:
            extras[index] += 1
        return {
            record_key(item): int(base + extra)
            for item, extra in zip(visible, extras)
        }
    def sync_event_actors(
        self,
        observations,
        visible_keys: set[str],
        center_ra: float,
        center_dec: float,
        energy_band: tuple[float, float],
        color_mode: str,
        point_size: float,
        opacity: float,
        rgb_config,
        density_strength: float,
        total_budget: int,
        representation: str = "events",
        reset_camera: bool = False,
        render: bool = True,
    ):
        if not self.available:
            return
        actors = self._density_actors if representation == "density" else self._event_actors
        signatures = self._density_signatures if representation == "density" else self._event_signatures
        observations = list(observations)
        existing = {record_key(obs) for obs in observations}
        for key in list(actors):
            if key not in existing:
                actor = actors.pop(key)
                self._plotter.remove_actor(actor, render=False)
                signatures.pop(key, None)
        budgets = self.display_budgets(observations, visible_keys, total_budget)
        low, high = energy_band
        for obs in observations:
            key = record_key(obs)
            actor = actors.get(key)
            if key not in visible_keys:
                if actor is not None:
                    self._set_actor_visible(actor, False)
                continue
            budget = budgets.get(key, 1)
            signature = (
                round(low, 4),
                round(high, 4),
                color_mode,
                tuple(round(x, 4) for x in rgb_config[0]),
                tuple(round(x, 4) for x in rgb_config[1]),
                tuple(round(x, 4) for x in rgb_config[2]),
                round(float(rgb_config[3]), 3),
                round(float(rgb_config[4]), 3),
                round(density_strength, 3),
                budget,
                round(center_ra, 6),
                round(center_dec, 6),
            )
            if signatures.get(key) != signature:
                frame = obs.frame
                if not frame.empty:
                    energy = frame["KEV"].to_numpy(dtype=np.float32, copy=False)
                    frame = frame.loc[(energy >= low) & (energy <= high)]
                local = _sample_frame(frame, budget).reset_index(drop=True)
                self._build_or_update_event_actor(
                    key,
                    local,
                    center_ra,
                    center_dec,
                    color_mode,
                    point_size,
                    opacity,
                    rgb_config,
                    density_strength,
                    mission=str(obs.record.mission),
                    actors=actors,
                    pickable=representation == "events",
                )
                signatures[key] = signature
            else:
                self._set_point_actor_appearance(actor, point_size, opacity)
            actor = actors.get(key)
            if actor is not None:
                self._set_actor_visible(actor, True)
        for actor in self._voxel_actors.values():
            self._set_actor_visible(actor, False)
        for actor in (self._density_actors if representation == "events" else self._event_actors).values():
            self._set_actor_visible(actor, False)
        self._ensure_axes()
        if reset_camera:
            self._plotter.reset_camera()
        if render:
            self._plotter.render()
    def publish_observation_actor(
        self, observation, center_ra, center_dec, energy_band, color_mode,
        point_size, opacity, rgb_config, density_strength, budget,
        *, reset_camera=False,
    ):
        if not self.available:
            return
        low, high = energy_band
        frame = observation.frame
        if not frame.empty:
            energy = frame["KEV"].to_numpy(dtype=np.float32, copy=False)
            frame = frame.loc[(energy >= low) & (energy <= high)]
        local = _sample_frame(frame, max(1, int(budget))).reset_index(drop=True)
        key = record_key(observation)
        self._build_or_update_event_actor(
            key, local, center_ra, center_dec, color_mode, point_size, opacity,
            rgb_config, density_strength, mission=str(observation.record.mission),
        )
        self._event_signatures.pop(key, None)
        self._ensure_axes()
        if reset_camera:
            self._plotter.reset_camera()
        self._plotter.render()
    def _build_or_update_event_actor(
        self, key, frame, center_ra, center_dec, color_mode, point_size,
        opacity, rgb_config, density_strength, mission="", actors=None, pickable=True,
    ):
        actors = self._event_actors if actors is None else actors
        if frame.empty:
            actor = actors.pop(key, None)
            if actor is not None:
                self._plotter.remove_actor(actor, render=False)
            if pickable:
                self._pick_frames[key] = frame
                self._pick_points[key] = np.empty((0, 3), dtype=np.float32)
            return
        points = self._display_coordinates(frame, center_ra, center_dec)
        finite = np.all(np.isfinite(points), axis=1)
        local = frame.loc[finite].reset_index(drop=True)
        points = points[finite]
        if color_mode == "mission":
            missions = (
                local["MISSION"].astype(str).to_numpy()
                if "MISSION" in local
                else np.full(len(local), str(mission), dtype=object)
            )
            colors = mission_event_colors(missions)
        elif color_mode == "rgb":
            colors = rgb_event_colors(
                local["KEV"].to_numpy(dtype=np.float32, copy=False),
                rgb_config[0],
                rgb_config[1],
                rgb_config[2],
                rgb_config[3],
                rgb_config[4],
            )
        else:
            colors = energy_gradient_colors(local["KEV"].to_numpy(dtype=np.float32, copy=False))
        if density_strength > 0 and len(local) <= 70_000:
            density = point_density(local)
            factor = np.clip(0.35 + 0.75 * density_strength * density, 0.18, 1.0)
            colors = np.asarray(np.clip(colors * factor[:, None], 0, 255), dtype=np.uint8)
        cloud = pv.PolyData(points)
        cloud["RGB"] = colors
        actor = actors.get(key)
        if actor is None or not self._update_point_actor(actor, cloud, point_size, opacity):
            if actor is not None:
                self._plotter.remove_actor(actor, render=False)
            actor = self._plotter.add_points(
                cloud,
                scalars="RGB",
                rgb=True,
                point_size=float(point_size),
                opacity=float(opacity),
                render_points_as_spheres=False,
                name=f"events::{key}",
                reset_camera=False,
            )
            actors[key] = actor
        if pickable:
            self._pick_frames[key], self._pick_points[key] = local, points
    def _remove_event_actor(self, key: str):
        actor = self._event_actors.pop(key, None)
        if actor is not None:
            self._plotter.remove_actor(actor, render=False)
        self._event_signatures.pop(key, None)
        self._pick_frames.pop(key, None)
        self._pick_points.pop(key, None)
    def sync_voxel_actors(
        self,
        observations,
        visible_keys: set[str],
        center_ra: float,
        center_dec: float,
        energy_band: tuple[float, float],
        spatial_voxel: float,
        energy_voxel: float,
        smooth_spatial: float,
        smooth_energy: float,
        threshold_fraction: float,
        max_cells: int,
        opacity: float = 0.82,
        show_edges: bool = True,
        reset_camera: bool = False,
        render: bool = True,
    ):
        if not self.available:
            return
        low, high = energy_band
        for actor in self._event_actors.values():
            self._set_actor_visible(actor, False)
        for obs in observations:
            key = record_key(obs)
            if key not in visible_keys:
                actor = self._voxel_actors.get(key)
                if actor is not None:
                    self._set_actor_visible(actor, False)
                continue
            signature = (
                round(low, 4),
                round(high, 4),
                round(spatial_voxel, 4),
                round(energy_voxel, 4),
                round(smooth_spatial, 4),
                round(smooth_energy, 4),
                round(threshold_fraction, 4),
                int(max_cells),
                round(center_ra, 6),
                round(center_dec, 6),
            )
            if self._voxel_signatures.get(key) != signature:
                frame = obs.frame
                energy = frame["KEV"].to_numpy(dtype=np.float32, copy=False)
                frame = frame.loc[(energy >= low) & (energy <= high)]
                product_signature = (*signature[:6], signature[7], *signature[-2:])
                product = self._voxel_products.get(key)
                if self._voxel_product_signatures.get(key) != product_signature:
                    product = voxel_histogram(
                        frame, center_ra, center_dec, spatial_voxel, energy_voxel,
                        smooth_spatial=smooth_spatial, smooth_energy=smooth_energy, max_cells=max_cells,
                    )
                    self._voxel_products[key] = product
                    self._voxel_product_signatures[key] = product_signature
                actor = self._build_voxel_actor(
                    key,
                    product,
                    threshold_fraction,
                    opacity,
                    _edges_for_grid(show_edges, product),
                    show_scalar_bar=not self._voxel_actors,
                )
                if actor is not None:
                    self._voxel_actors[key] = actor
                    self._voxel_signatures[key] = signature
                else:
                    self._voxel_actors.pop(key, None)
                    self._voxel_signatures.pop(key, None)
            actor = self._voxel_actors.get(key)
            if actor is not None:
                self._set_actor_visible(actor, True)
                set_voxel_actor_appearance(
                    actor, opacity, _edges_for_grid(
                        show_edges, self._voxel_products.get(key)
                    )
                )
        self._ensure_axes()
        if reset_camera:
            self._plotter.reset_camera()
        if render:
            self._plotter.render()
    def _build_voxel_actor(
        self,
        key,
        product,
        threshold_fraction,
        opacity,
        show_edges,
        show_scalar_bar=False,
    ):
        old = self._voxel_actors.get(key)
        if old is not None:
            self._plotter.remove_actor(old, render=False)
        if product is None:
            return None
        hist, edges = product
        if not np.any(hist > 0):
            return None
        hist, spacing, origin, _z_centers, energy_centers = transformed_voxel_geometry(hist, edges, self._scene_transform)
        grid = pv.ImageData(dimensions=np.asarray(hist.shape) + 1, spacing=spacing, origin=origin)
        grid.cell_data["Density"] = hist.flatten(order="F")
        grid.cell_data["Energy"] = np.broadcast_to(energy_centers[None, None, :], hist.shape).flatten(order="F")
        threshold = float(np.max(hist)) * float(threshold_fraction)
        voxels = grid.threshold(threshold, scalars="Density")
        # Match the readable vtk.js style: preserve exact histogram cells but
        # leave a narrow air gap so adjacent voxels remain visually distinct.
        if 0 < voxels.n_cells <= 120_000:
            try:
                voxels = voxels.shrink(0.94)
            except Exception:
                pass
        return self._plotter.add_mesh(
            voxels,
            scalars="Energy",
            cmap="turbo_r",
            opacity=float(opacity),
            show_edges=bool(show_edges),
            edge_color="#334155",
            line_width=0.20,
            smooth_shading=False,
            ambient=0.38,
            diffuse=0.62,
            specular=0.10,
            specular_power=12.0,
            show_scalar_bar=bool(show_scalar_bar),
            scalar_bar_args={
                "title": "Energy (keV)", "color": PLOT_TEXT,
                "title_font_size": 11, "label_font_size": 9,
                "n_labels": 4, "fmt": "%.3f", "vertical": True,
                "position_x": 0.88, "position_y": 0.22,
                "width": 0.07, "height": 0.48,
            } if show_scalar_bar else None,
            name=f"voxels::{key}",
            reset_camera=False,
        )
    def set_record_visibility(self, key: str, visible: bool, mode: str, render: bool = True):
        actors = self._actors_for_mode(mode)
        actor = actors.get(key)
        if actor is not None:
            self._set_actor_visible(actor, visible)
            if render and self.available:
                self._plotter.render()
    def set_records_visibility(self, keys, visible: bool, mode: str, render: bool = True):
        for key in keys:
            self.set_record_visibility(str(key), visible, mode, render=False)
        if render and self.available:
            self._plotter.render()
    def has_record_actor(self, key: str, mode: str) -> bool:
        return key in self._actors_for_mode(mode)
    def _actors_for_mode(self, mode):
        if mode == "density":
            return self._density_actors
        if mode == "voxels":
            return self._voxel_actors
        return self._event_actors
    def set_representation(self, mode, render=True):
        for current, actors in (("events", self._event_actors), ("density", self._density_actors), ("voxels", self._voxel_actors)):
            for actor in actors.values():
                self._set_actor_visible(actor, current == mode)
        if render and self.available:
            self._plotter.render()
    def remove_record(self, key: str, render: bool = True):
        if not self.available:
            return
        self._remove_event_actor(key)
        actor = self._voxel_actors.pop(key, None)
        if actor is not None:
            self._plotter.remove_actor(actor, render=False)
        actor = self._spectral_actors.pop(key, None)
        if actor is not None:
            self._plotter.remove_actor(actor, render=False)
        self._voxel_signatures.pop(key, None)
        self._voxel_products.pop(key, None)
        self._voxel_product_signatures.pop(key, None)
        actor = self._density_actors.pop(key, None)
        if actor is not None:
            self._plotter.remove_actor(actor, render=False)
        self._density_signatures.pop(key, None)
        self._slice_point_signatures.clear()
        for actor in self._slice_point_actors.values():
            self._set_actor_visible(actor, False)
        if render:
            self._plotter.render()
    @staticmethod
    def _set_actor_visible(actor, visible: bool):
        try:
            actor.SetVisibility(bool(visible))
        except Exception:
            try:
                actor.visibility = bool(visible)
            except Exception:
                pass
    def sync_slices(self, slices, region, center_dec: float = 0.0, show_planes: bool = True, render: bool = True):
        if not self.available or region is None:
            return
        wanted = {item.uid for item in slices if show_planes and item.visible and item.show_plane}
        for uid in list(self._slice_actors):
            if uid not in wanted:
                self._plotter.remove_actor(self._slice_actors.pop(uid), render=False)
                self._slice_signatures.pop(uid, None)
        half_x = region.radius_deg
        half_y = region.radius_deg
        for item in slices:
            if not show_planes or not item.visible or not item.show_plane:
                continue
            signature = (
                round(item.center_kev, 5),
                round(item.opacity, 3),
                item.color,
                round(half_x, 4),
                round(half_y, 4),
            )
            if self._slice_signatures.get(item.uid) == signature:
                continue
            old = self._slice_actors.get(item.uid)
            if old is not None:
                self._plotter.remove_actor(old, render=False)
            transform = self._scene_transform or EnergySceneTransform(
                float(region.center_ra_deg), float(center_dec), 0.0, 1.0
            )
            if transform.absolute_coordinates:
                viewport = SkyViewport.from_region(region)
                plane_center = (
                    transform.center_ra_deg,
                    transform.center_dec_deg,
                    float(transform.energy_to_scene(item.center_kev)),
                )
                plane_i_size = max(viewport.ra_max_deg - viewport.ra_min_deg, 1e-4)
                plane_j_size = max(viewport.dec_max_deg - viewport.dec_min_deg, 1e-4)
            else:
                plane_center = (0.0, 0.0, float(transform.energy_to_scene(item.center_kev)))
                plane_i_size = max(2 * half_x * 60.0, 1e-4)
                plane_j_size = max(2 * half_y * 60.0, 1e-4)
            plane = pv.Plane(
                center=plane_center,
                direction=(0, 0, 1),
                i_size=plane_i_size,
                j_size=plane_j_size,
            )
            actor = exclude_actor_from_bounds(self._plotter.add_mesh(
                plane,
                color=item.color,
                opacity=item.opacity,
                lighting=False,
                name=f"slice::{item.uid}",
                reset_camera=False,
            ))
            self._slice_actors[item.uid] = actor
            self._slice_signatures[item.uid] = signature
        if render:
            self._plotter.render()
    def sync_reference_plane(self, region, reference_kev, visible=True, render=True):
        if not self.available or region is None:
            return
        reference_plane(self, region, reference_kev, visible)
        if render:
            self._plotter.render()
    def sync_slice_points(
        self, slices, observations, visible_keys, center_ra, center_dec,
        point_size, content_mode="all", active_uid=None, total_budget=300_000,
        render=True,
    ):
        if not self.available:
            return
        enabled = enabled_slice_point_uids(slices, content_mode, active_uid)
        for uid in list(self._slice_point_actors):
            if uid not in enabled:
                self._set_actor_visible(self._slice_point_actors[uid], False)
        budgets = self.display_budgets(observations, set(visible_keys), total_budget)
        for item in slices:
            if item.uid not in enabled:
                continue
            parts = []
            for obs in observations:
                key = record_key(obs)
                if key not in visible_keys:
                    continue
                frame = obs.frame
                if frame.empty:
                    continue
                values = frame["KEV"].to_numpy(dtype=float, copy=False)
                local = frame.loc[(values >= item.low_kev) & (values <= item.high_kev)]
                parts.append(_sample_frame(local, budgets.get(key, 1)))
            frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            signature = (round(item.low_kev, 5), round(item.high_kev, 5), item.color, len(frame),
                         round(center_ra, 6), round(center_dec, 6))
            actor = self._slice_point_actors.get(item.uid)
            if self._slice_point_signatures.get(item.uid) != signature:
                points = self._display_coordinates(frame, center_ra, center_dec) if not frame.empty else np.empty((0, 3), dtype=np.float32)
                if actor is not None:
                    self._plotter.remove_actor(actor, render=False)
                if len(points):
                    cloud = pv.PolyData(points)
                    actor = self._plotter.add_points(
                        cloud, color=item.color, point_size=float(point_size) * 1.2,
                        opacity=min(1.0, float(item.opacity) + 0.25),
                        render_points_as_spheres=False, name=f"slice-points::{item.uid}",
                        reset_camera=False,
                    )
                    self._slice_point_actors[item.uid] = actor
                else:
                    self._slice_point_actors.pop(item.uid, None)
                self._slice_point_signatures[item.uid] = signature
            actor = self._slice_point_actors.get(item.uid)
            if actor is not None:
                self._set_point_actor_appearance(actor, float(point_size) * 1.2, min(1.0, float(item.opacity) + 0.25))
                self._set_actor_visible(actor, True)
        if render:
            self._plotter.render()
    def set_content_mode(self, mode: str, visible_keys, active_uid=None, render=True):
        if not self.available:
            return
        show_normal = mode not in ("active", "planes")
        visible_keys = set(visible_keys)
        for key, actor in self._event_actors.items():
            self._set_actor_visible(actor, show_normal and key in visible_keys)
        if render:
            self._plotter.render()
    def set_top_image(self, product, region, mode="off", opacity=0.75, palette="gray",
                      stretch="log", brightness=1.0, contrast=1.0, render=True):
        if not self.available or region is None:
            return
        try:
            from jaxa_udon3.desktop.top_image import build_top_image
            build_top_image(self, product, region, mode, opacity, palette, stretch, brightness, contrast)
        except Exception:
            if self._top_image_actor is not None:
                self._set_actor_visible(self._top_image_actor, False)
        if render:
            self._plotter.render()
    def _picked(self, point):
        from jaxa_udon3.desktop.picking import pick_event
        pick_event(self, point)
    def reset(self):
        if self.available:
            self._plotter.reset_camera()
    def screenshot(self, path: str):
        if self.available:
            self._plotter.screenshot(str(path))
