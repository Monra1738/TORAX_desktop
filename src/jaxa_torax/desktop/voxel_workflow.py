from __future__ import annotations


def resolved_voxel_energy_band(state) -> tuple[float, float]:
    """Return the range that drives voxel construction, independent of plane visibility."""
    source = getattr(state, "voxel_energy_source", "selected_slice")
    selected = state.selected_slice()
    if source == "selected_slice" and selected is not None:
        return float(selected.low_kev), float(selected.high_kev)
    if source == "all_energies":
        ranges = []
        for observation in state.loaded_observations:
            frame = observation.frame
            if not frame.empty and "KEV" in frame:
                ranges.append((float(frame.KEV.min()), float(frame.KEV.max())))
        if ranges:
            return min(low for low, _ in ranges), max(high for _, high in ranges)
    return tuple(map(float, state.energy_band))
