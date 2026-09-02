import numpy as np
import pandas as pd


def pick_event(view, point):
    if point is None:
        return
    p = np.asarray(point, dtype=float)
    if p.shape != (3,):
        return
    best = None
    for key, points in view._pick_points.items():
        actor = view._event_actors.get(key)
        if points.size == 0 or actor is None:
            continue
        try:
            if not actor.GetVisibility():
                continue
        except Exception:
            pass
        idx = int(np.argmin(np.sum((points - p) ** 2, axis=1)))
        distance = float(np.sum((points[idx] - p) ** 2))
        if best is None or distance < best[0]:
            best = (distance, key, idx)
    if best is not None:
        frame = view._pick_frames.get(best[1], pd.DataFrame())
        if 0 <= best[2] < len(frame):
            view.event_selected.emit(frame.iloc[best[2]].to_dict())
