"""Customised ViserPlayViewer with camera-state preservation across re-renders.

Background
----------
viser's ``ViserServer.on_client_connect`` (viser/_viser.py) not only queues the
callback for *future* connections — it also immediately submits it to a thread
pool for every *already-connected* client.  This means that when
``ViserPlayViewer.setup()`` calls ``create_scene_gui()``, which registers an
``on_client_connect`` handler that resets the camera to a computed default
offset, that reset fires on the live browser session.

``CameraRestoringViserViewer`` overrides ``setup()`` to capture the saved
camera state and, once the parent's handler has been submitted, spawn a short
daemon thread (50 ms delay) that restores the saved position.  The 50 ms gap
is far longer than the sub-millisecond WebSocket round-trip needed for the
camera-reset message, so by the time we restore the camera is settled.

``server.initial_camera`` is also updated so that any *new* connections (e.g.
a browser tab refresh) open at the same saved angle.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import viser
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.viser.viewer import ViserPlayViewer

_TRAINING_METRIC_PREFIXES = ("Episode_Metrics/", "Train/mean_reward", "Train/mean_episode_length")
_TRAINING_METRICS_UPDATE_INTERVAL_S = 30


def _read_tfevents_history(log_dir: str) -> dict[str, tuple]:
    """Return {tag: (steps, values)} for display-relevant scalars in log_dir.

    Uses size_guidance=0 (unlimited) so the full training history is returned.
    Each ScalarEvent has .step (iteration) and .value (float).
    Tags with fewer than 2 events are omitted (uplot needs at least 2 points).
    """
    import numpy as np
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    result = {}
    for tag in ea.Tags().get("scalars", []):
        if any(tag.startswith(p) for p in _TRAINING_METRIC_PREFIXES):
            events = ea.Scalars(tag)
            if len(events) >= 2:
                result[tag] = (
                    np.array([e.step for e in events], dtype=np.float64),
                    np.array([e.value for e in events], dtype=np.float64),
                )
    return result


class CameraRestoringViserViewer(ViserPlayViewer):
    """ViserPlayViewer that preserves the user's camera angle across re-renders.

    Pass ``post_setup_camera`` as a ``{"position": np.ndarray, "look_at":
    np.ndarray}`` dict captured from the connected client just before the
    previous viewer was torn down.  If ``None`` (first render), the default
    camera configured via ``env_cfg.viewer`` is used unchanged.
    """

    def __init__(
        self,
        *args: Any,
        post_setup_camera: dict | None = None,
        **kwargs: Any,
    ) -> None:
        self._post_setup_camera = post_setup_camera
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        super().setup()

        if self._post_setup_camera is None:
            return

        camera_state = self._post_setup_camera
        server = self._server

        # Update initial_camera now (synchronous, no race) so future new
        # connections open at the saved angle.
        try:
            server.initial_camera.position = camera_state["position"]
            server.initial_camera.look_at = camera_state["look_at"]
        except Exception:
            pass

        # super().setup() registered an on_client_connect handler that resets
        # the camera for existing clients via the server's thread pool.
        # We wait 50 ms (>> WebSocket round-trip) so that message has been
        # sent, then push the restore on top.
        def _delayed_restore() -> None:
            time.sleep(0.05)
            for client in server.get_clients().values():
                try:
                    client.camera.position = camera_state["position"]
                    client.camera.look_at = camera_state["look_at"]
                except Exception:
                    pass

        threading.Thread(target=_delayed_restore, daemon=True).start()


_POLL_INTERVAL_S = 15


class PollingViserViewer(CameraRestoringViserViewer):
    """CameraRestoringViserViewer that polls for new checkpoints in the background.

    Every _POLL_INTERVAL_S seconds the checkpoint manager's fetch_available() is
    called.  When previously-unseen checkpoints appear a green "Load: <name>" button
    appears below the checkpoint dropdown in the Checkpoints tab.  Notifications are
    removed individually: clicking a button loads that specific checkpoint; selecting
    a checkpoint from the dropdown removes its notification if one exists.  No bulk
    clearing happens.
    """

    def __init__(self, *args: Any, log_dir: str | None = None, **kwargs: Any) -> None:
        self._log_dir = log_dir
        self._training_metrics_md: Any = None
        self._training_metrics_tab_id: str | None = None
        self._training_metric_plots: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        super().setup()
        if self._ckpt_mgr is not None:
            # Store the Checkpoints tab container ID so _add_notification() can
            # inject new elements into the same tab dynamically from the poll thread.
            self._ckpt_tab_container_id = self._ckpt_dropdown._impl.parent_container_id
            # Keyed by checkpoint filename so we can remove by name.
            self._ckpt_notification_handles: dict[str, Any] = {}
            # When the user selects from the dropdown, remove only the matching button.
            @self._ckpt_dropdown.on_update
            def _(_: Any) -> None:
                if self._ckpt_user_event.is_set():
                    name = self._ckpt_dropdown.value.split("  (")[0]
                    self._remove_notification_for(name)
            threading.Thread(target=self._poll_loop, daemon=True).start()

        # Standalone "Training Metrics" tab group — only shown in render_training_result mode.
        if self._log_dir is not None:
            with self._server.gui.add_tab_group().add_tab(
                "Training Metrics", icon=viser.Icon.CHART_BAR
            ):
                self._training_metrics_tab_id = self._server.gui._get_container_uuid()
                self._training_metrics_md = self._server.gui.add_markdown(
                    "_Awaiting first metrics update (≤30s)…_"
                )
            threading.Thread(target=self._training_metrics_loop, daemon=True).start()

    # ── training metrics panel ────────────────────────────────────────────────

    def _training_metrics_loop(self) -> None:
        """Read the TFEvents log every 30 s and update per-metric uplot history panels."""
        while True:
            time.sleep(_TRAINING_METRICS_UPDATE_INTERVAL_S)
            if self._log_dir is None or self._training_metrics_tab_id is None:
                continue
            try:
                history = _read_tfevents_history(self._log_dir)
            except Exception:
                continue
            if not history:
                continue

            # Remove placeholder markdown on first data arrival.
            if self._training_metrics_md is not None:
                try:
                    self._training_metrics_md.remove()
                except Exception:
                    pass
                self._training_metrics_md = None

            for tag, (steps, values) in sorted(history.items()):
                label = tag.split("/", 1)[-1]
                if tag not in self._training_metric_plots:
                    try:
                        prev_id = self._server.gui._get_container_uuid()
                        self._server.gui._set_container_uuid(self._training_metrics_tab_id)
                        handle = self._server.gui.add_uplot(
                            data=(steps, values),
                            series=(
                                viser.uplot.Series(label=""),
                                viser.uplot.Series(label=label),
                            ),
                            title=label,
                            aspect=0.5,
                        )
                        self._server.gui._set_container_uuid(prev_id)
                        self._training_metric_plots[tag] = handle
                    except Exception:
                        pass
                else:
                    try:
                        self._training_metric_plots[tag].data = (steps, values)
                    except Exception:
                        pass

    # ── background polling ────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        known: set[str] = set()
        first = True
        while True:
            time.sleep(_POLL_INTERVAL_S)
            try:
                entries = self._ckpt_mgr.fetch_available()
            except Exception:
                continue
            names = {n for n, _ in entries}
            new = names - known
            if not first and new:
                for checkpoint_name in sorted(new):
                    self._add_notification(checkpoint_name)
                self._actions.append((ViewerAction.FETCH_CHECKPOINT, "latest_silent"))
            known = names
            first = False

    # ── per-checkpoint notification buttons ──────────────────────────────────

    def _add_notification(self, checkpoint_name: str) -> None:
        """Add a load button into the Checkpoints tab for a newly available checkpoint."""
        if checkpoint_name in self._ckpt_notification_handles:
            return  # already shown
        try:
            prev_id = self._server.gui._get_container_uuid()
            self._server.gui._set_container_uuid(self._ckpt_tab_container_id)
            btn = self._server.gui.add_button(
                f"Load: {checkpoint_name}",
                color="green",
                hint="Click to load this checkpoint",
            )
            self._ckpt_notification_handles[checkpoint_name] = btn
            self._server.gui._set_container_uuid(prev_id)

            @btn.on_click
            def _(_: Any) -> None:
                self._load_checkpoint_from_notification(checkpoint_name)
        except Exception:
            pass

    def _remove_notification_for(self, checkpoint_name: str) -> None:
        """Remove the notification button for one specific checkpoint, if present."""
        btn = self._ckpt_notification_handles.pop(checkpoint_name, None)
        if btn is not None:
            try:
                btn.remove()
            except Exception:
                pass

    def _load_checkpoint_from_notification(self, checkpoint_name: str) -> None:
        """Remove this notification and load the checkpoint directly.

        Loads inline rather than via the action queue so that the parent's
        "selected" handler never calls ``dropdown.value = cur`` while
        ``_ckpt_user_event`` is set — which would fire ``on_update`` and
        accidentally remove other notification buttons.
        """
        self._remove_notification_for(checkpoint_name)
        if self._ckpt_mgr is None:
            return
        labels = list(self._ckpt_dropdown.options)
        matching = next(
            (lbl for lbl in labels if lbl.startswith(checkpoint_name)),
            checkpoint_name,
        )
        # Keep user_event cleared during the value assignment so the synchronous
        # on_update callback does not remove any other notification buttons.
        self._ckpt_user_event.clear()
        self._ckpt_dropdown.value = matching
        self._ckpt_user_event.set()
        if checkpoint_name != self._ckpt_mgr.current_name:
            print(f"[INFO]: Loading {checkpoint_name}...")
            self.policy = self._ckpt_mgr.load_checkpoint(checkpoint_name)
            self._ckpt_mgr.current_name = checkpoint_name
            self.reset_environment()
            print(f"[INFO]: Loaded {checkpoint_name}")

    # ── action handler ────────────────────────────────────────────────────────

    def _handle_custom_action(self, action: Any, payload: Any) -> bool:
        # "Use Latest" button: after super() loads, remove that checkpoint's notification.
        if action == ViewerAction.FETCH_CHECKPOINT and payload == "latest":
            result = super()._handle_custom_action(action, payload)
            self._remove_notification_for(self._ckpt_mgr.current_name)
            return result

        if action != ViewerAction.FETCH_CHECKPOINT or payload != "latest_silent":
            return super()._handle_custom_action(action, payload)

        if self._ckpt_mgr is None:
            return True
        entries = self._ckpt_mgr.fetch_available()
        if not entries:
            return True
        labels = [f"{n}  ({t})" if t else n for n, t in entries]
        # The displayed label may have a stale timestamp string, so find the
        # updated label for the same checkpoint name and restore it explicitly.
        # This prevents viser from resetting the dropdown to options[0] when the
        # current value is no longer in the new options list.
        displayed_name = self._ckpt_dropdown.value.split("  (")[0]
        restored_label = next(
            (lbl for lbl in labels if lbl.startswith(displayed_name)),
            self._ckpt_dropdown.value,
        )
        self._ckpt_user_event.clear()
        self._ckpt_dropdown.options = labels
        self._ckpt_dropdown.value = restored_label
        self._ckpt_user_event.set()
        # No auto-loading: user must click a notification button or select from dropdown.
        return True
