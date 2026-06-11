import os
from pathlib import Path as _Path

import gradio as gr
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.envs import ManagerBasedRlEnv
import numpy as np
from datetime import datetime

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

if gr.NO_RELOAD:

    import json
    import time
    import threading
    from dataclasses import asdict, dataclass, field
    from pathlib import Path
    from typing import List

    import sys as _sys

    import torch
    import viser
    from gradio_rangeslider import RangeSlider
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.viewer.viser.viewer import ViserPlayViewer
    from myosuite.core.myouser_hydra_cli import load_config_interactive
    from myosuite.envs.myo.backends.mjlab.register_mjlab_myouser_tasks import (
        _universal_spec_fn,
        _get_myosuite_root,
        register_mjlab_myouser_task,
    )
    import time as _time
    from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago
    from myosuite.utils.onnx_checkpoint import is_onnx_checkpoint_name, onnx_checkpoint_sort_key
    from myosuite.integrations.musclemimic.mjlab_policy_runner import OnnxCheckpointingMjlabRunner

    # viser_viewer_ext.py lives alongside gradio_ui.py.
    _this_dir = str(Path(__file__).resolve().parent)
    if _this_dir not in _sys.path:
        _sys.path.insert(0, _this_dir)
    from viser_viewer_ext import CameraRestoringViserViewer, PollingViserViewer

    myouser_path = Path(__file__).resolve().parent  # gradio_ui.py IS already inside myouser/

    _viser_preview_server = viser.ViserServer(port=7861)
    try:
        _viser_preview_url = _viser_preview_server.request_share_url()
    except Exception:
        _viser_preview_url = f"http://localhost:{_viser_preview_server.get_port()}"

    _render_state: dict = {
        "env": None,
        "vec_env": None,
        "viewer": None,
        "stop_event": None,
        "thread": None,
        "struct_key": None,         # (model_path, num_targets, target_shape_tuple)
        "policy": None,             # OnnxPolicy instance, or None for zero-action
        "checkpoint_manager": None, # CheckpointManager for viser checkpoint tab
    }

    def _render_struct_key(config):
        """Return a hashable key that captures the structural aspects of the config.

        A key change means the MuJoCo model must be rebuilt from scratch.
        Non-structural changes (colors, sizes, position ranges) can be applied
        to the running env without restarting.

        All BMParameters fields require a rebuild:
          model_path  — different XML skeleton
          ctrl_dt     — changes env decimation
          reset_type  — changes events dict
          sigdepnoise/constantnoise type — changes action-term noise config
          fatigue_enabled — changes whether TorchFatigueState is created
          mj_impl     — changes MuJoCo backend
        """
        targets = config.env.task_config.targets
        num_t = targets.num_targets
        shapes = tuple(
            getattr(targets, f"target_{i}").name for i in range(num_t)
        )
        noise = config.env.muscle_config.noise_params
        return (
            config.env.model_path,
            config.env.ctrl_dt,
            config.env.task_config.reset_type,
            noise.sigdepnoise_type,
            noise.constantnoise_type,
            config.env.muscle_config.fatigue_enabled,
            config.env.mj_impl,
            num_t,
            shapes,
        )

    # Hardcoded target coordinate origin (matches register_mjlab_myouser_tasks.py line ~542).
    _TARGET_ORIGIN = [-0.0068, -0.1747, 1.0257]
    _UNIVERSAL_ENTITY = "universal_robot"

    def _update_model_geoms(env, config) -> None:
        """Update mj_model geom rgba/size and event term position/size ranges.

        Architecture notes
        ------------------
        Viser (mjviser) bakes colors and sizes into trimesh vertex colors when
        ViserPlayViewer.setup() calls create_primitive_mesh() / _resolve_flat_rgba().
        This reads mj_model.geom_rgba and mj_model.geom_size — NOT the WARP batched
        model (env.sim.model). There is no per-frame color/size sync from WARP.

        dr.geom_rgba / dr.geom_size only update env.sim.model (WARP tensors,
        initialized from mj_model via mjwarp.put_model() but independent thereafter).
        Calling those DR callbacks would not affect viser.

        Correct fix: update mj_model directly, then restart ViserPlayViewer so
        setup() re-reads mj_model and rebuilds the scene handles with new values.

        Geom IDs come from asset.target_size_ids (the same global IDs used by the
        TaskEntity and DR terms) to avoid any entity-namespace ambiguity that
        mujoco.mj_name2id would have.
        """
        mj_model = env.sim.mj_model  # standard mujoco.MjModel — what viser renders from
        targets = config.env.task_config.targets
        num_active = targets.num_targets

        try:
            asset = env.scene[_UNIVERSAL_ENTITY]
            geom_ids_tensor = asset.target_size_ids  # shape [num_targets], global geom IDs
        except Exception:
            geom_ids_tensor = None

        pos_ranges: dict = {}
        size_ranges: dict = {}

        for i in range(num_active):
            target_cfg = getattr(targets, f"target_{i}")

            # Update mj_model so the restarted ViserPlayViewer sees new colors/sizes.
            if geom_ids_tensor is not None and i < len(geom_ids_tensor):
                geom_id = int(geom_ids_tensor[i].item())
                rgb = list(target_cfg.rgb)
                mj_model.geom_rgba[geom_id] = [rgb[0], rgb[1], rgb[2], 1.0]

                size_range = target_cfg.size
                if isinstance(size_range[0], (int, float)):
                    # Sphere: scalar [min_r, max_r] → uniform radius
                    avg_r = float((size_range[0] + size_range[1]) / 2)
                    mj_model.geom_size[geom_id] = [avg_r, avg_r, avg_r]
                else:
                    # Box: [[min_x,min_y,min_z], [max_x,max_y,max_z]] → per-axis half-extents
                    avg_size = [
                        float((size_range[0][k] + size_range[1][k]) / 2)
                        for k in range(3)
                    ]
                    mj_model.geom_size[geom_id, :3] = avg_size

            # Update event term params so env.reset() samples new position/size ranges.
            pos_ranges[f"body_target_{i}"] = {
                xyz: (
                    target_cfg.position[0][xyz] + _TARGET_ORIGIN[xyz],
                    target_cfg.position[1][xyz] + _TARGET_ORIGIN[xyz],
                )
                for xyz in range(3)
            }
            s = target_cfg.size
            size_ranges[f"geom_target_{i}"] = {
                xyz: (
                    s[0] if isinstance(s[0], (int, float)) else s[0][xyz],
                    s[1] if isinstance(s[1], (int, float)) else s[1][xyz],
                )
                for xyz in range(3)
            }

        try:
            env.event_manager.get_term_cfg("target_pos_dr").params["ranges"] = pos_ranges
        except Exception:
            pass
        try:
            env.event_manager.get_term_cfg("target_size_dr").params["ranges"] = size_ranges
        except Exception:
            pass

    def _build_policy_and_ckpt_manager(checkpoint_run: str, checkpoint_file: str, device: str):
        """Build (policy, CheckpointManager) from dropdown selections.

        Uses OnnxCheckpointingMjlabRunner (PyTorch actor) exactly as play_notebook_viser.py.
        The runner is initialised from the current gradio vec_env; if obs_dim does not
        match the checkpoint, load_onnx raises and (None, None) is returned.

        Returns (None, None) when no checkpoint is selected or loading fails.
        """
        ckpt_path = checkpoint_path_from_run_filename(checkpoint_run, checkpoint_file)
        if not ckpt_path or ckpt_path == "None":
            return None, None
        p = Path(ckpt_path)
        if p.suffix != ".onnx" or not p.exists():
            print(f"[render] Checkpoint not found or not .onnx: {p}")
            return None, None
        vec_env = _render_state.get("vec_env")
        if vec_env is None:
            print(f"[render] No vec_env available; cannot load checkpoint.")
            return None, None
        try:
            rl_cfg = load_rl_cfg("myoUserUniversal-v0")
            runner = OnnxCheckpointingMjlabRunner(
                vec_env, asdict(rl_cfg), str(p.parent), device, task_id="myoUserUniversal-v0"
            )
            runner.load_onnx(p)
            policy = runner.get_inference_policy(device=device)
            print(f"[render] Loaded checkpoint from {p.name}")

            ckpt_dir = p.parent

            def _reload_policy(path: str):
                runner.load_onnx(Path(path))
                return runner.get_inference_policy(device=device)

            def _fetch_local() -> list[tuple[str, str]]:
                now = _time.time()
                entries: list[tuple[str, str]] = []
                for f in ckpt_dir.glob("*.onnx"):
                    if not is_onnx_checkpoint_name(f.name):
                        continue
                    ago = format_time_ago(int(now - f.stat().st_mtime))
                    entries.append((f.name, ago))
                entries.sort(key=lambda x: onnx_checkpoint_sort_key(x[0]))
                return entries

            ckpt_manager = CheckpointManager(
                current_name=p.name,
                fetch_available=_fetch_local,
                load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
            )
            return policy, ckpt_manager
        except Exception as exc:
            import traceback
            print(f"[render] Could not load checkpoint from {p}: {exc}")
            traceback.print_exc()
            return None, None

    class _StopViewer(BaseException):
        """Raised from the policy callback to cleanly unwind ViserPlayViewer.run()."""

    class _StoppablePolicy:
        """Wraps any callable policy and raises _StopViewer when stop_event is set."""
        def __init__(self, inner, stop_event: threading.Event) -> None:
            self._inner = inner
            self._stop = stop_event

        def __call__(self, obs):
            if self._stop.is_set():
                raise _StopViewer
            return self._inner(obs)

        def reset(self) -> None:
            if hasattr(self._inner, "reset"):
                self._inner.reset()

    class _ZeroActionInner:
        """Stateless zero-action policy (no stop logic — that lives in _StoppablePolicy)."""
        def __init__(self, num_actions: int, device: str) -> None:
            self._n = num_actions
            self._dev = device

        def __call__(self, obs):
            n = next(iter(obs.values())).shape[0] if hasattr(obs, "values") else obs.shape[0]
            return torch.zeros(n, self._n, device=self._dev)

        def reset(self) -> None:
            pass

    def _run_viewer(viewer: ViserPlayViewer, stop_event: threading.Event) -> None:
        try:
            viewer.run()
        except _StopViewer:
            pass
        except Exception:
            pass  # env closed externally


sphere_ranges = {
    "x": (0.2, 0.55),
    "y": (-0.25, 0.45),
    "z": (-0.3, 0.3),
    "size": (0.001, 0.15),  # radius
}

INIT_ELEMENTS = 2


parent_path = myouser_path.parent
CHECKPOINT_PATH = os.path.join(parent_path, "tracked_checkpoints/universal/")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
LOG_DIR = os.path.join(parent_path, "logs/rsl_rl/myo_universal")
os.makedirs(LOG_DIR, exist_ok=True)


@dataclass
class RunState:
    wandb_run_name: str = ""
    cfg_overrides: List[str] = field(default_factory=lambda: [])
    log_dir: str = ""          # set by training cell; used by render_training_result()
    is_training: bool = False  # set True during runner.learn(); blocks render operations


def checkpoints_diff_folder(show_training_ckpts: bool = False):
    tracked_checkpoints = os.listdir(CHECKPOINT_PATH)
    if show_training_ckpts:
        tracked_checkpoints += os.listdir(LOG_DIR)
    return tracked_checkpoints


def get_available_checkpoints(show_training_ckpts: bool = False):
    tracked_checkpoints = checkpoints_diff_folder(show_training_ckpts=show_training_ckpts)
    checkpoints = tracked_checkpoints + ["None"]
    return checkpoints


def is_number(s: str):
    if ("." not in s) and s.isdigit():
        return True
    return False


def _run_checkpoint_dir(checkpoint_run: str) -> str | None:
    """Return the directory containing .onnx files for a run.

    Accepted layout:
    - ``CHECKPOINT_PATH/<run>/``
    Returns None when this directory does not exist.
    """
    base = os.path.join(CHECKPOINT_PATH, checkpoint_run)
    for sub in ("",):
        candidate = os.path.join(base, sub)
        if os.path.isdir(candidate):
            return candidate
    return None


def get_available_checkpoint_numbers(checkpoint_run):
    if checkpoint_run == "None":
        return ["None"]
    tracked_checkpoints = checkpoints_diff_folder()
    if checkpoint_run not in tracked_checkpoints:
        return ["None"]
    checkpoint_path = _run_checkpoint_dir(checkpoint_run)
    if checkpoint_path is None:
        return ["None"]
    available_numbers = [
        f for f in os.listdir(checkpoint_path)
        if os.path.isfile(os.path.join(checkpoint_path, f))
        and os.path.splitext(f)[1] in (".onnx", ".pt")
    ]
    available_numbers.sort(key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]) if "model_final" not in x else float("inf"), reverse=True)
    return available_numbers


def checkpoint_path_from_run_filename(checkpoint_run, checkpoint_filename):
    if checkpoint_run == "None" or checkpoint_filename == "None":
        return "None"
    tracked_checkpoints = checkpoints_diff_folder()
    if checkpoint_run not in tracked_checkpoints:
        return "None"
    ckpt_dir = _run_checkpoint_dir(checkpoint_run)
    if ckpt_dir is None:
        return "None"
    return os.path.join(ckpt_dir, checkpoint_filename)


def update_checkpoint_dirs(show_training_ckpts):
    choices = get_available_checkpoints(show_training_ckpts)
    return gr.update(choices=choices, value=choices[0])


def update_checkpoint_numbers(checkpoint_run):
    choices = get_available_checkpoint_numbers(checkpoint_run)
    return gr.update(choices=choices, value=choices[0])


def extract_rgb(rgba):
    rgba = rgba.split("rgba(")[1].strip(")")
    r, g, b, a = rgba.split(",")
    r = r.strip()
    g = g.strip()
    b = b.strip()
    rgb = [float(x.strip()) / 255.0 for x in [r, g, b]]
    return rgb


def hex_to_rgb(hex_color):
    if "rgba" in hex_color:
        return extract_rgb(hex_color)
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return rgb


def init_target_rgb(i):
    target_rgb = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.8, 0.8, 0.1],
        [0.4, 0.4, 0.4],
        [0.4, 0.1, 0.4],
        [0.1, 0.4, 0.4],
        [0.4, 0.4, 0.1],
        [0.1, 0.1, 0.1],
        [0.8, 0.8, 0.8],
    ]
    rgbs = [el * 255 for el in target_rgb[i]]
    return f"rgba({rgbs[0]},{rgbs[1]},{rgbs[2]},1)"


def sanity_check_float(value, name):
    try:
        float(value)
    except Exception as e:
        gr.Warning(f"{name} must be a float, got {value}")
        return False
    return True


def sanity_check_bool(value, name):
    try:
        bool(value)
    except Exception as e:
        gr.Warning(f"{name} must be a boolean, got {value}")
        return False
    return True


def sanity_check_choices(value, name, choices):
    try:
        assert value in choices
    except Exception as e:
        gr.Warning(
            f"{name} must be a string and one of the following: {choices} but got {value}"
        )
        return False
    return True


def sanity_check_rgb(rgb, name):
    len_rgb = len(rgb)
    if len_rgb != 3:
        gr.Warning(f"RGB {name} must be a list of 3 floats, got {rgb}")
        return False
    for i, el in enumerate(rgb):
        if not sanity_check_float(el, f" RGB {name} element {i}"):
            return False
    return True


def sanity_check_int(value, name):
    try:
        int(value)
    except Exception as e:
        gr.Warning(f"{name} must be an integer, got {value}")
        return False
    return True


def sanity_check_number_targets(num_targets, name):
    try:
        assert 1 <= int(num_targets) <= 10
    except Exception as e:
        gr.Warning(
            f"Number of targetsmust be an integer between 1 and 10 but got {num_targets}"
        )
        return False
    return True


def sanity_check_float_array(array, name):
    try:
        to_array = np.array(array)
        for i, el in enumerate(to_array):
            assert sanity_check_float(el, f"{name} element {i}")
    except Exception as e:
        print(e)
        gr.Warning(f"{name} must be an array of floats, got {array}")
        return False
    return True


class BMParameters:
    num_elements: int = 7
    reset_type_choices = ("zero", "epsilon_uniform", "range_uniform", "None")
    mj_impl_choices = ("mjlab",)
    bm_model_choices = ("MoBL_Arms_Index", "MoBL_Arms_Hand", "MyoArm_nohand", "MyoArm")
    
    @staticmethod
    def fields():
        return [
            "bm_model",
            "ctrl_dt",
            "reset_type",
            "sigdepnoise_enabled",
            "constantnoise_enabled",
            "fatigue_enabled",
            "mj_impl_type",
        ]

    @staticmethod
    def get_parameters():
        with gr.Row():
            bm_model = gr.Dropdown(
                label="Biomechanical Model",
                choices=BMParameters.bm_model_choices,
                interactive=True,
                value="MoBL_Arms_Index",
            )
            ctrl_dt = gr.Number(
                label="Control Timestep (s)",
                value=0.05,
                minimum=0.01,
                maximum=0.5,
                step=0.01,
                interactive=True,
            )
            reset_type = gr.Dropdown(
                label="Define how to reset the body pose",
                choices=BMParameters.reset_type_choices,
                interactive=True,
                value="epsilon_uniform",
            )
            sigdepnoise_enabled = gr.Checkbox(
                label="Signal-dependent motor noise", value=False, interactive=True
            )
            constantnoise_enabled = gr.Checkbox(
                label="Constant motor noise", value=False, interactive=True
            )
            fatigue_enabled = gr.Checkbox(
                label="Muscle Fatigue", value=False, interactive=True
            )
            mj_impl_type = gr.Dropdown(
                label="MuJoCo Backend",
                choices=BMParameters.mj_impl_choices,
                interactive=True,
                value="mjlab",
                visible="hidden",
            )
        return bm_model, ctrl_dt, reset_type, sigdepnoise_enabled, constantnoise_enabled, fatigue_enabled, mj_impl_type

    @staticmethod
    def get_my_args(all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = 10
        box_number = 10 * BoxParameters.num_elements
        sphere_number = 10 * SphereParameters.num_elements
        bm_start = rl_number + num_targets + radio_number + box_number + sphere_number
        bm_end = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + BMParameters.num_elements
        )
        return all_args[bm_start:bm_end]

    @staticmethod
    def sanity_check_args(
        bm_model, ctrl_dt, reset_type, sigdepnoise_enabled, constantnoise_enabled, fatigue_enabled, mj_impl_type
    ):
        sanity_check = sanity_check_choices(
            bm_model, "Biomechanical Model", BMParameters.bm_model_choices
        )
        sanity_check &= sanity_check_float(ctrl_dt, "Ctrl Timestep (s)")
        sanity_check &= sanity_check_choices(
            reset_type, "Reset type", BMParameters.reset_type_choices
        )
        sanity_check &= sanity_check_bool(
            sigdepnoise_enabled, "Signal-dependent motor noise"
        )
        sanity_check &= sanity_check_bool(constantnoise_enabled, "Constant motor noise")
        sanity_check &= sanity_check_bool(fatigue_enabled, "Muscle Fatigue")
        sanity_check &= sanity_check_choices(
            mj_impl_type, "MuJoCo Implementation", BMParameters.mj_impl_choices
        )
        return sanity_check

    @classmethod
    def parse_values(cls, all_args):
        bm_model, ctrl_dt, reset_type, sigdepnoise_enabled, constantnoise_enabled, fatigue_enabled, mj_impl_type = (
            cls.get_my_args(all_args)
        )
        sanity_check = cls.sanity_check_args(
            bm_model, ctrl_dt, reset_type, sigdepnoise_enabled, constantnoise_enabled, fatigue_enabled, mj_impl_type
        )
        if not sanity_check:
            return False, []
        overrides = []
        ## TODO: move logic (argument to config attribute mappings) to separate function
        overrides.append(f"env.ctrl_dt={float(ctrl_dt)}")
        overrides.append(f"env.task_config.reset_type={reset_type}")
        sigdepnoise_type = "white" if sigdepnoise_enabled else "None"
        constantnoise_type = "white" if constantnoise_enabled else "None"
        overrides.append(
            f"env.muscle_config.noise_params.sigdepnoise_type={sigdepnoise_type}"
        )
        overrides.append(
            f"env.muscle_config.noise_params.constantnoise_type={constantnoise_type}"
        )
        overrides.append(f"env.muscle_config.fatigue_enabled={fatigue_enabled}")
        overrides.append(f"env.mj_impl={mj_impl_type}")
        if bm_model == "MoBL_Arms_Index":
            bm_model_path = "myosuite/envs/myo/assets/arm/mobl_arms_index_universal_myouser.xml"
            ee_site = "fingertip"
            ref_site = "humphant"
        elif bm_model == "MoBL_Arms_Hand":
            bm_model_path = "myosuite/envs/myo/assets/arm/mobl_arms_hand_universal_myouser.xml"
            ee_site = "IFtip"
            ref_site = "humphant"
        elif bm_model == "MyoArm_nohand":
            bm_model_path = "myosuite/envs/myo/assets/arm/myoarm_nohand_universal_myouser.xml"
            ee_site = "IFtip"
            ref_site = "R.Shoulder_marker"
        elif bm_model == "MyoArm":
            bm_model_path = "myosuite/envs/myo/assets/arm/myoarm_universal_myouser.xml"
            ee_site = "IFtip"
            ref_site = "R.Shoulder_marker"
        else:
            raise ValueError(f"Invalid biomechanical model: {bm_model}")
        overrides.append(f"env.model_path={bm_model_path}")
        overrides.append(f"env.task_config.reach_settings.ref_site={ref_site}")
        overrides.append(f"env.task_config.reach_settings.ee_site={ee_site}")
        return True, overrides


class BoxParameters:
    num_elements: int = 8

    @staticmethod
    def init_button_positions(i):
        button_positions = [
            [0.42, 0.02, 0.01],
            [0.42, 0.32, 0.01],
            [0.42, 0.02, -0.29],
            [0.42, 0.32, -0.29],
            [0.5, 0.0, -0.05],
            [0.5, 0.07, -0.05],
            [0.41, -0.07, 0.05],
            [0.41, 0.07, 0.05],
            [0.41, -0.07, -0.25],
            [0.41, 0.0, -0.25],
        ]
        return button_positions[i]

    @staticmethod
    def fields():
        return [
            "box_position_x",
            "box_position_y",
            "box_position_z",
            "box_size_x",
            "box_size_y",
            "min_touch_force",
            "orientation_angle",
            "rgb",
        ]

    @staticmethod
    def get_parameters(i):
        # Boxes option row
        button_position = BoxParameters.init_button_positions(i)
        button_rgb = init_target_rgb(i)
        with gr.Row(visible=False) as box_row:
            with gr.Accordion(label=f"Box {i+1} Settings", open=False):
                with gr.Row():
                    box_position_x = RangeSlider(
                        label="Depth Position",
                        minimum=0.2,
                        maximum=0.55,
                        value=(button_position[0], button_position[0]),
                        step=0.001,
                        interactive=True,
                    )
                    box_position_y = RangeSlider(
                        label="Horizontal Position",
                        minimum=-0.25,
                        maximum=0.45,
                        value=(button_position[1], button_position[1]),
                        step=0.001,
                        interactive=True,
                    )
                    box_position_z = RangeSlider(
                        label="Vertical Position",
                        minimum=-0.3,
                        maximum=0.3,
                        value=(button_position[2], button_position[2]),
                        step=0.001,
                        interactive=True,
                    )
                with gr.Row():
                    box_size_x_slider = RangeSlider(
                        label="Width",
                        minimum=0.005,
                        maximum=0.06,
                        value=(0.05, 0.05),
                        step=0.001,
                        interactive=True,
                    )
                    box_size_y_slider = RangeSlider(
                        label="Height",
                        minimum=0.005,
                        maximum=0.06,
                        value=(0.05, 0.05),
                        step=0.001,
                        interactive=True,
                    )
                with gr.Row():
                    min_touch_force = gr.Slider(
                        label="Minimum Touch Force",
                        info="Minimum force required to register a touch",
                        minimum=0.0,
                        maximum=10.0,
                        value=1.0,
                        step=0.1,
                        interactive=True,
                    )
                    orientation_angle = gr.Slider(
                        label="Orientation Angle",
                        minimum=0.0,
                        maximum=180.0,
                        value=45,
                        step=1.0,
                        interactive=True,
                    )
                    rgb_btn = gr.ColorPicker(
                        label="RGB", value=button_rgb, interactive=True
                    )
        return box_row, (
            box_position_x,
            box_position_y,
            box_position_z,
            box_size_x_slider,
            box_size_y_slider,
            min_touch_force,
            orientation_angle,
            rgb_btn,
        )

    @staticmethod
    def get_my_args(i, all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = 10
        box_start = rl_number + radio_number + num_targets
        box_end = box_start + 10 * BoxParameters.num_elements
        box_args = all_args[box_start:box_end]
        start_index = i * BoxParameters.num_elements
        end_index = start_index + BoxParameters.num_elements
        return box_args[start_index:end_index]

    @staticmethod
    def sanity_check_args(
        i,
        box_position_x,
        box_position_y,
        box_position_z,
        box_size_x_slider,
        box_size_y_slider,
        min_touch_force,
        orientation_angle,
        rgb,
    ):
        sanity_check = sanity_check_float(box_position_x[0], f"Box {i+1} Position X Range Start")
        sanity_check &= sanity_check_float(box_position_x[1], f"Box {i+1} Position X Range End")
        sanity_check &= sanity_check_float(box_position_y[0], f"Box {i+1} Position Y Range Start")
        sanity_check &= sanity_check_float(box_position_y[1], f"Box {i+1} Position Y Range End")
        sanity_check &= sanity_check_float(box_position_z[0], f"Box {i+1} Position Z Range Start")
        sanity_check &= sanity_check_float(box_position_z[1], f"Box {i+1} Position Z Range End")
        sanity_check &= sanity_check_float(box_size_x_slider[0], f"Box {i+1} Size X Range Start")
        sanity_check &= sanity_check_float(box_size_x_slider[1], f"Box {i+1} Size X Range End")
        sanity_check &= sanity_check_float(box_size_y_slider[0], f"Box {i+1} Size Y Range Start")
        sanity_check &= sanity_check_float(box_size_y_slider[1], f"Box {i+1} Size Y Range End")
        sanity_check &= sanity_check_float(
            min_touch_force, f"Box {i+1} Min Touch Force"
        )
        sanity_check &= sanity_check_float(
            orientation_angle, f"Box {i+1} Orientation Angle"
        )
        sanity_check &= sanity_check_rgb(rgb, f"Box {i+1} RGB")
        return sanity_check

    @classmethod
    def parse_values(cls, i, all_args):
        (
            box_position_x,
            box_position_y,
            box_position_z,
            box_size_x_slider,
            box_size_y_slider,
            min_touch_force,
            orientation_angle,
            rgb_btn,
        ) = cls.get_my_args(i, all_args)
        try:
            rgb = hex_to_rgb(rgb_btn)
        except Exception as e:
            gr.Warning(f"Box {i+1} RGB must be a valid hex color code, got {rgb_btn}")
            return False, []
        sanity_check = cls.sanity_check_args(
            i,
            box_position_x,
            box_position_y,
            box_position_z,
            box_size_x_slider,
            box_size_y_slider,
            min_touch_force,
            orientation_angle,
            rgb,
        )
        if not sanity_check:
            return False, []
        overrides = []
        overrides.append(
            f"env.task_config.targets.target_{i}.position=[[{box_position_x[0]},{box_position_y[0]},{box_position_z[0]}],[{box_position_x[1]},{box_position_y[1]},{box_position_z[1]}]]"
        )
        overrides.append(
            f"env.task_config.targets.target_{i}.size=[[{0.5*box_size_x_slider[0]},{0.5*box_size_y_slider[0]},0.01],[{0.5*box_size_x_slider[1]},{0.5*box_size_y_slider[1]},0.01]]"
        )
        overrides.append(
            f"env.task_config.targets.target_{i}.min_touch_force={min_touch_force}"
        )
        overrides.append(
            f"env.task_config.targets.target_{i}.euler=[0,{-orientation_angle*np.pi/180},0]"
        )
        overrides.append(
            f"env.task_config.targets.target_{i}.rgb=[{rgb[0]},{rgb[1]},{rgb[2]}]"
        )
        return True, overrides


class SphereParameters:
    num_elements: int = 6

    @staticmethod
    def fields():
        return ["x_range", "y_range", "z_range", "size_range", "dwell_duration", "rgb"]

    @staticmethod
    def get_parameters(i, dwell_duration_min=0.0):

        # Sphere option row
        sphere_rgb = init_target_rgb(i)
        with gr.Row(visible=True) as sphere_row:
            with gr.Accordion(label=f"Sphere {i+1} Settings", open=False):
                with gr.Row():
                    gr.Markdown(
                        "#### The coordinates for the sphere targets are randomly sampled from a range, please choose them below"
                    )
                with gr.Row():
                    x_slider = RangeSlider(
                        label=f"Depth Range",
                        minimum=sphere_ranges["x"][0],
                        maximum=sphere_ranges["x"][1],
                        value=(sphere_ranges["x"][0], sphere_ranges["x"][1]),
                        step=0.001,
                        interactive=True,
                    )
                    y_slider = RangeSlider(
                        label=f"Horizontal Range",
                        minimum=sphere_ranges["y"][0],
                        maximum=sphere_ranges["y"][1],
                        value=(sphere_ranges["y"][0], sphere_ranges["y"][1]),
                        step=0.001,
                        interactive=True,
                    )
                    z_slider = RangeSlider(
                        label=f"Vertical Range",
                        minimum=sphere_ranges["z"][0],
                        maximum=sphere_ranges["z"][1],
                        value=(sphere_ranges["z"][0], sphere_ranges["z"][1]),
                        step=0.001,
                        interactive=True,
                    )
                with gr.Row():
                    size_slider = RangeSlider(
                        label=f"Size Range (Sphere Radius)",
                        minimum=sphere_ranges["size"][0],
                        maximum=sphere_ranges["size"][1],
                        value=(sphere_ranges["size"][0], sphere_ranges["size"][1]),
                        step=0.001,
                        interactive=True,
                    )
                    dwell_duration = gr.Number(
                        label=f"Dwell Duration",
                        value=0.25,
                        minimum=dwell_duration_min,
                        maximum=1.0,
                        step=0.01,
                        interactive=True,
                    )
                    color_picker = gr.ColorPicker(
                        label=f"RGB", value=sphere_rgb, interactive=True
                    )
        return sphere_row, (
            x_slider,
            y_slider,
            z_slider,
            size_slider,
            dwell_duration,
            color_picker,
        )

    def get_my_args(i, all_args):
        rl_number = RLParameters.num_elements
        radio_number = 10
        box_number = 10 * BoxParameters.num_elements
        num_targets = 1
        sphere_start = rl_number + radio_number + box_number + num_targets
        sphere_args = all_args[sphere_start:]
        start_index = i * SphereParameters.num_elements
        end_index = start_index + SphereParameters.num_elements
        return sphere_args[start_index:end_index]

    @staticmethod
    def sanity_check_args(
        i, x_range, y_range, z_range, size_range, dwell_duration, rgb
    ):
        sanity_check = sanity_check_float(x_range[0], f"Sphere {i+1} X Range Start")
        sanity_check &= sanity_check_float(x_range[1], f"Sphere {i+1} X Range End")
        sanity_check &= sanity_check_float(y_range[0], f"Sphere {i+1} Y Range Start")
        sanity_check &= sanity_check_float(y_range[1], f"Sphere {i+1} Y Range End")
        sanity_check &= sanity_check_float(z_range[0], f"Sphere {i+1} Z Range Start")
        sanity_check &= sanity_check_float(z_range[1], f"Sphere {i+1} Z Range End")
        sanity_check &= sanity_check_float(
            size_range[0], f"Sphere {i+1} Size Range Start"
        )
        sanity_check &= sanity_check_float(
            size_range[1], f"Sphere {i+1} Size Range End"
        )
        sanity_check &= sanity_check_float(
            dwell_duration, f"Sphere {i+1} Dwell Duration"
        )
        sanity_check &= sanity_check_rgb(rgb, f"Sphere {i+1} RGB")
        return sanity_check

    @classmethod
    def parse_values(cls, i, all_args):
        x_range, y_range, z_range, size_range, dwell_duration, rgb = cls.get_my_args(
            i, all_args
        )
        try:
            rgb = hex_to_rgb(rgb)
        except Exception as e:
            gr.Warning(f"Sphere {i+1} RGB must be a valid hex color code, got {rgb}")
            return False, []
        sanity_check = cls.sanity_check_args(
            i, x_range, y_range, z_range, size_range, dwell_duration, rgb
        )
        if not sanity_check:
            return False, []
        overrides = []
        overrides.append(
            f"env.task_config.targets.target_{i}.position=[[{x_range[0]},{y_range[0]},{z_range[0]}],[{x_range[1]},{y_range[1]},{z_range[1]}]]"
        )
        overrides.append(
            f"env.task_config.targets.target_{i}.size=[{size_range[0]},{size_range[1]}]"
        )
        overrides.append(
            f"env.task_config.targets.target_{i}.dwell_duration={dwell_duration}"
        )
        overrides.append(
            f"env.task_config.targets.target_{i}.rgb=[{rgb[0]},{rgb[1]},{rgb[2]}]"
        )
        return True, overrides


class TaskParameters:
    num_elements: int = 1

    @staticmethod
    def fields():
        return ["max_duration"]

    @staticmethod
    def get_parameters(ctrl_dt=0.05):
        with gr.Row():
            max_duration = gr.Number(
                label="Maximum Episode Duration (s)",
                value=4.0,
                minimum=0.5,
                maximum=120.0,
                step=ctrl_dt,
                interactive=True,
            )
        return (max_duration,)

    @staticmethod
    def get_my_args(all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = 10
        box_number = 10 * BoxParameters.num_elements
        sphere_number = 10 * SphereParameters.num_elements
        bm_number = BMParameters.num_elements
        task_start = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
        )
        task_end = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + TaskParameters.num_elements
        )
        return all_args[task_start:task_end]

    @classmethod
    def parse_values(cls, all_args):
        (max_duration,) = cls.get_my_args(all_args)
        sanity_check = sanity_check_float(max_duration, "Maximum Episode Duration (s)")
        if not sanity_check:
            return False, []
        overrides = []
        overrides.append(f"env.task_config.max_duration={float(max_duration)}")
        return True, overrides


class ObservationSpace:
    num_elements: int = 4

    possible_obs_keys = ["qpos", "qvel", "qacc", "ee_pos", "act"]
    possible_fatigue_obs_keys = ["MA", "MR", "MF"]
    possible_vision_keys = ["rgb", "depth"]
    possible_omni_keys = ["target_pos", "target_size", "phase_progress", "dwell_fraction"]

    @classmethod
    def fields(cls):
        return [*cls.possible_obs_keys, *cls.possible_fatigue_obs_keys, *cls.possible_vision_keys, *cls.possible_omni_keys]

    @staticmethod
    def get_parameters():
        obs_keys = []
        with gr.Row():
            for k in ObservationSpace.possible_obs_keys:
                obs_key = gr.Checkbox(label=k, value=True, interactive=True)
                obs_keys.append(obs_key)

        fatigue_obs_keys = []
        with gr.Row():
            for k in ObservationSpace.possible_fatigue_obs_keys:
                fatigue_obs_key = gr.Checkbox(label=k, value=False, interactive=False)
                fatigue_obs_keys.append(fatigue_obs_key)

        vision_keys = []
        with gr.Row():
            for k in ObservationSpace.possible_vision_keys:
                vision_key = gr.Checkbox(label=k, value=False, interactive=True)
                vision_keys.append(vision_key)

        omni_keys = []
        with gr.Row():
            for k in ObservationSpace.possible_omni_keys:
                omni_key = gr.Checkbox(label=k, value=True, interactive=True)
                omni_keys.append(omni_key)

        return obs_keys, fatigue_obs_keys, vision_keys, omni_keys

    @staticmethod
    def get_my_args(all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = 10
        box_number = 10 * BoxParameters.num_elements
        sphere_number = 10 * SphereParameters.num_elements
        bm_number = BMParameters.num_elements
        task_number = TaskParameters.num_elements
        obs_start = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + task_number
        )
        obs_end = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + task_number
            + len(ObservationSpace.possible_obs_keys)
        )
        fatigue_obs_start = obs_end
        fatigue_obs_end = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + task_number
            + len(ObservationSpace.possible_obs_keys)
            + len(ObservationSpace.possible_fatigue_obs_keys)
        )
        vision_start = fatigue_obs_end
        vision_end = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + task_number
            + len(ObservationSpace.possible_obs_keys)
            + len(ObservationSpace.possible_fatigue_obs_keys)
            + len(ObservationSpace.possible_vision_keys)
        )
        omni_start = vision_end
        omni_end = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + task_number
            + len(ObservationSpace.possible_obs_keys)
            + len(ObservationSpace.possible_fatigue_obs_keys)
            + len(ObservationSpace.possible_vision_keys)
            + len(ObservationSpace.possible_omni_keys)
        )
        return all_args[obs_start:obs_end], all_args[fatigue_obs_start:fatigue_obs_end], all_args[vision_start:vision_end], all_args[omni_start:omni_end]
    @classmethod
    def parse_values(cls, all_args):
        obs_keys, fatigue_obs_keys, vision_keys, omni_keys = cls.get_my_args(all_args)
        obs_keys_selected = [
            ObservationSpace.possible_obs_keys[id] for id, k in enumerate(obs_keys) if k
        ]  # [k.label for k in obs_keys if k.value]
        fatigue_obs_keys_selected = [
            ObservationSpace.possible_fatigue_obs_keys[id] for id, k in enumerate(fatigue_obs_keys) if k
        ]  # [k.label for k in fatigue_obs_keys if k.value]
        vision_keys_selected = [
            ObservationSpace.possible_vision_keys[id] for id, k in enumerate(vision_keys) if k
        ]
        vision_enabled = len(vision_keys_selected) > 0
        vision_enabled_flag = "enabled" if vision_enabled else "disabled"
        vision_mode = None
        if "rgb" in vision_keys_selected:
            if "depth" in vision_keys_selected:
                vision_mode = "rgbd"
            else:
                vision_mode = "rgb"
        elif "depth" in vision_keys_selected:
            vision_mode = "depth"
        omni_keys_selected = [
            ObservationSpace.possible_omni_keys[id]
            for id, k in enumerate(omni_keys)
            if k
        ]  # [k.label for k in omni_keys if k.value]
        overrides = []
        overrides.append(f"env.task_config.obs_keys={obs_keys_selected}")
        overrides.append(f"env.muscle_config.fatigue_obs_keys={fatigue_obs_keys_selected}")
        overrides.append(f"vision={vision_enabled_flag}")
        if vision_enabled:
            overrides.append(f"vision.vision_mode={vision_mode}")
        overrides.append(f"env.task_config.omni_keys={omni_keys_selected}")
        return True, overrides


class RewardFunction:
    num_elements: int = 1

    weights_default_min_max_step = {
        "distance": (1, 0, 10, 0.1),
        "subtask_bonus": (0, 0, 10, 0.5),
        "completion_bonus": (10, 0, 50, 1),
        "neural_effort": (0, 0, 1, 0.01),
    }
    keys = ["distance", "subtask_bonus", "completion_bonus", "neural_effort"]

    @staticmethod
    def fields():
        return [*RewardFunction.keys]
    
    @staticmethod
    def get_parameters():
        reward_weights = []
        with gr.Row():
            for k, (
                value,
                minimum,
                maximum,
                step,
            ) in RewardFunction.weights_default_min_max_step.items():
                reward_weight = gr.Number(
                    label=k,
                    value=value,
                    minimum=minimum,
                    maximum=maximum,
                    step=step,
                    interactive=True,
                )
                reward_weights.append(reward_weight)

        return (reward_weights,)

    @staticmethod
    def get_my_args(all_args):
        rl_number = RLParameters.num_elements
        num_targets = 1
        radio_number = 10
        box_number = 10 * BoxParameters.num_elements
        sphere_number = 10 * SphereParameters.num_elements
        bm_number = BMParameters.num_elements
        task_number = TaskParameters.num_elements
        obs_number = len(ObservationSpace.possible_obs_keys) + len(
            ObservationSpace.possible_fatigue_obs_keys) + len(
            ObservationSpace.possible_vision_keys) + len(
            ObservationSpace.possible_omni_keys
        )
        reward_start = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + task_number
            + obs_number
        )
        reward_end = (
            rl_number
            + num_targets
            + radio_number
            + box_number
            + sphere_number
            + bm_number
            + task_number
            + obs_number
            + len(RewardFunction.weights_default_min_max_step)
        )
        return all_args[reward_start:reward_end]

    @staticmethod
    def sanity_check_args(reward_weights):
        sanity_check = True
        for i, k in enumerate(reward_weights):
            key = RewardFunction.keys[i]
            sanity_check &= sanity_check_float(k, f"Reward Weight {key}")
        return sanity_check

    @classmethod
    def parse_values(cls, all_args):
        reward_weights = cls.get_my_args(all_args)
        sanity_check = cls.sanity_check_args(reward_weights)
        if not sanity_check:
            return False, []
        overrides = []
        for i, k in enumerate(reward_weights):
            key = RewardFunction.keys[i]
            rename_keys = {
                "distance": "distance",
                "subtask_bonus": "phase_bonus",
                "completion_bonus": "done",
            }
            key = rename_keys.get(key, key)
            overrides.append(f"env.task_config.weighted_reward_keys.{key}={k}")
        return True, overrides


class RLParameters:
    num_elements: int = 10
    num_mds: int = 1
    
    @staticmethod
    def fields():
        return [
        "Number of Total Training Steps",
        "Number of Checkpoints During/After Training",
        "Number of Evaluations During/After Training",
        "Batch Size",
        "Number of Parallel Environments",
        "Number of Minibatches",
        "Select Experiment from Which to Load Checkpoints",
        "Select Checkpoint Number",
        "Show Checkpoints from Own Training Runs",
        "Target Initial Seed",
    ]

    @staticmethod
    def get_parameters():
        with gr.Row():
            num_timesteps = gr.Number(
                label="Number of Total Training Steps",
                value=15000000,
                minimum=0,
                maximum=100000000,
                step=100_000,
                interactive=True,
            )
            checkpoint_interval = gr.Number(
                label="Checkpoint Interval (in Number of Training Iterations)",
                value=30,
                minimum=1,
                maximum=1000,
                step=10,
                interactive=True,
            )
        md_rl_note = gr.Markdown(
            "<span style='font-size: 1em;'>"
            "<b><span style='color:red'>Note:</span></b> If GPU memory errors occur, try with a smaller <i>number of parallel environments</i>."
            "</span>",
            elem_id="hint-text",
        )
        with gr.Row():
            num_envs = gr.Number(
                label="Number of Parallel Environments",
                value=1024,
                minimum=0,
                maximum=4096,
                interactive=True,
            )
            unroll_length = gr.Number(
                label="Unroll Length", value=10, minimum=0, maximum=200, interactive=True
            )
            num_minibatches = gr.Number(
                label="Number of Minibatches",
                value=8,
                minimum=1,
                maximum=40,
                interactive=True,
            )
            num_epochs_per_update = gr.Number(
                label="Number of Epochs Per Update",
                value=8,
                minimum=1,
                maximum=50,
                interactive=True,
            )
        show_training_ckpts = gr.Checkbox(
            label="Show Checkpoints from Own Training Runs",
            value=False,
            interactive=True,
            visible=False,
        )
        with gr.Row():
            choices = get_available_checkpoints(show_training_ckpts.value)
            select_checkpoint_run = gr.Dropdown(
                label="Select Experiment from Which to Load Checkpoints",
                choices=choices,
                interactive=True,
                value="None",
                allow_custom_value=True,
            )
            choices = get_available_checkpoint_numbers(select_checkpoint_run.value)
            select_checkpoint_file = gr.Dropdown(
                label="Select Checkpoint File",
                choices=choices,
                interactive=True,
                value="None",
                allow_custom_value=True,
            )
            select_checkpoint_run.change(
                update_checkpoint_numbers,
                inputs=select_checkpoint_run,
                outputs=select_checkpoint_file,
            )
        show_training_ckpts.change(
            update_checkpoint_dirs,
            inputs=show_training_ckpts,
            outputs=select_checkpoint_run,
        )     
        target_init_seed = gr.Number(
            label="Target Initial Seed",
            value=0,
            minimum=0,
            maximum=1000000,
            interactive=True,
            visible=False,
        )
        return (
            num_timesteps,
            checkpoint_interval,
            num_envs,
            unroll_length,
            num_minibatches,
            num_epochs_per_update,
            select_checkpoint_run,
            select_checkpoint_file,
            show_training_ckpts,
            target_init_seed,
            md_rl_note,
        )

    @staticmethod
    def get_my_args(all_args):
        return all_args[: RLParameters.num_elements]

    @staticmethod
    def sanity_check_args(
        num_targets,
        num_timesteps,
        checkpoint_interval,
        num_envs,
        unroll_length,
        num_minibatches,
        num_epochs_per_update,
        select_checkpoint_run,
        select_checkpoint_file,
        show_training_ckpts,
        target_init_seed,
    ):
        sanity_check = sanity_check_number_targets(num_targets, "Number of Targets")
        sanity_check &= sanity_check_int(num_timesteps, "Number of Total Training Steps")
        sanity_check &= sanity_check_int(
            checkpoint_interval, "Checkpoint Interval (in Number of Training Iterations)"
        )
        sanity_check &= sanity_check_int(num_envs, "Number of Parallel Environments")
        sanity_check &= sanity_check_int(unroll_length, "Unroll Length")
        sanity_check &= sanity_check_int(num_minibatches, "Number of Minibatches")
        sanity_check &= sanity_check_int(num_epochs_per_update, "Number of Epochs Per Update")
        sanity_check &= sanity_check_choices(
            select_checkpoint_run,
            "Select Experiment from Which to Load Checkpoints",
            get_available_checkpoints(),
        )
        sanity_check &= sanity_check_choices(
            select_checkpoint_file,
            "Select Checkpoint File",
            get_available_checkpoint_numbers(select_checkpoint_run),
        )
        sanity_check &= sanity_check_bool(
            show_training_ckpts, "Show Checkpoints from Own Training Runs"
        )
        sanity_check &= sanity_check_int(target_init_seed, "Target Initial Seed")
        return sanity_check

    @classmethod
    def parse_values(cls, all_args):
        (
            num_timesteps,
            checkpoint_interval,
            num_envs,
            unroll_length,
            num_minibatches,
            num_epochs_per_update,
            select_checkpoint_run,
            select_checkpoint_file,
            show_training_ckpts,
            target_init_seed,
        ) = cls.get_my_args(all_args)
        overrides = []
        num_targets = cls.get_number_targets(all_args)
        sanity_check = cls.sanity_check_args(
            num_targets,
            num_timesteps,
            checkpoint_interval,
            num_envs,
            unroll_length,
            num_minibatches,
            num_epochs_per_update,
            select_checkpoint_run,
            select_checkpoint_file,
            show_training_ckpts,
            target_init_seed,
        )
        if not sanity_check:
            return False, []
        to_text = [
            "",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        ]
        overrides.append(f"env/task_config/targets={to_text[int(num_targets)]}")
        overrides.append(f"rl.num_timesteps={int(num_timesteps)}")
        overrides.append(f"rl.save_interval={int(checkpoint_interval)}")
        overrides.append(f"rl.num_envs={int(num_envs)}")
        overrides.append(f"rl.unroll_length={int(unroll_length)}")
        overrides.append(f"rl.num_minibatches={int(num_minibatches)}")
        overrides.append(f"rl.num_epochs_per_update={int(num_epochs_per_update)}")
        exact_checkpoint_path = checkpoint_path_from_run_filename(
            select_checkpoint_run, select_checkpoint_file
        )
        overrides.append(f"rl.load_checkpoint_path={exact_checkpoint_path}")
        overrides.append(f"env.task_config.target_init_seed={int(target_init_seed)}")
        return True, overrides

    @classmethod
    def get_number_targets(cls, all_args):
        return all_args[RLParameters.num_elements]

    @classmethod
    def get_target_init_seed(cls, all_args):
        return all_args[RLParameters.num_elements - 1]


def update_dwell_duration(dwell_duration, ctrl_dt):
    return gr.update(minimum=max(ctrl_dt, 0))


_CONFIG_DIR = _Path(__file__).resolve().parent / "gradio_configs"


class ConfigSaver:
    def check_folder(self):
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        files = os.listdir(_CONFIG_DIR)
        json_files = [file for file in files if file.endswith(".json")]
        file_names = [file.split(".")[0] for file in json_files]
        return file_names

    def __init__(self):
        self.my_configs = self.check_folder()
        self.run_inputs = None

    
    def to_labelled_dict(self, data):
        labelled_dict = {}
        rl_dict = dict(zip(RLParameters.fields(), RLParameters.get_my_args(data)))
        labelled_dict['rl'] = rl_dict
        labelled_dict['num_targets'] = data[RLParameters.num_elements]
        radio_start = RLParameters.num_elements + 1
        radio_end = radio_start + 10
        radio_keys = [f"target_{i+1}_radio" for i in range(10)]
        radio_values = data[radio_start:radio_end]
        # labelled_dict.update(zip(radio_keys, radio_values))
        for i in range(10):
            item_dict = {}
                        
            item_dict['radio'] = radio_values[i]
            item_dict['box'] = dict(zip(BoxParameters.fields(), BoxParameters.get_my_args(i, data)))
            item_dict['sphere'] = dict(zip(SphereParameters.fields(), SphereParameters.get_my_args(i, data)))
            labelled_dict[f"target_{i+1}"] = item_dict
        labelled_dict['bm'] = dict(zip(BMParameters.fields(), BMParameters.get_my_args(data)))
        labelled_dict['task'] = dict(zip(TaskParameters.fields(), TaskParameters.get_my_args(data)))
        labelled_dict['obs'] = dict(zip(ObservationSpace.fields(), ObservationSpace.get_my_args(data)))
        labelled_dict['reward'] = dict(zip(RewardFunction.fields(), RewardFunction.get_my_args(data)))
        return labelled_dict

    def from_labelled_dict(self, labelled_data):
        data = []
        data.extend(labelled_data['rl'].values())
        data.append(labelled_data['num_targets'])
        for i in range(10):
            data.append(labelled_data[f"target_{i+1}"]["radio"])
        for i in range(10):            
            data.extend(labelled_data[f"target_{i+1}"]["box"].values())
        for i in range(10):
            data.extend(labelled_data[f"target_{i+1}"]["sphere"].values())
        data.extend(labelled_data['bm'].values())
        data.extend(labelled_data['task'].values())
        obs_values = labelled_data['obs'].values()
        data.extend(sum(obs_values, []))
        reward_defaults = {k: v[0] for k, v in RewardFunction.weights_default_min_max_step.items()}
        data.extend(labelled_data['reward'].get(k, reward_defaults[k]) for k in RewardFunction.fields())
        return data

    def add_config(self, config_name, data):
        labelled_dict = self.to_labelled_dict(data)
        with open(_CONFIG_DIR / f"{config_name}.json", "w") as f:
            json.dump(labelled_dict, f, indent=4)

    def config_save_clicked(self, config_name_input, *args):
        if config_name_input == "":
            gr.Warning("Please enter a name for the configuration.")
        else:
            self.add_config(config_name_input, args)
            if config_name_input not in self.my_configs:
                self.my_configs.append(config_name_input)
        return gr.update(choices=self.my_configs, value=config_name_input)

    def available_configs(self):
        return self.my_configs

    def load_config(self, config_name):
        with open(_CONFIG_DIR / f"{config_name}.json", "r") as f:
            # data = json.load(f)
            labelled_data = json.load(f)
        data = self.from_labelled_dict(labelled_data)
        return tuple([gr.update(value=k) for k in data])


def load_test_environment():
    task_id = "myoUserUniversal-v0"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    play_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda", render_mode="rgb_array")

    return play_env


def get_ui(project_name, run_state=RunState(), use_legacy_rendering=False):
    # Fix box handlers properly
    with gr.Blocks() as demo:
        pageview = gr.Radio(
            choices=["Simple", "Advanced"],
            label=f"Show Advanced Options",
            value="Simple",
            interactive=True,
        )

        gr.Markdown("**Weights & Biases URL:**")
        url_display = gr.Textbox(
            value=f"https://wandb.ai/biom-rl-ui/{project_name}",
            label="Results Dashboard",
            interactive=False,
            show_copy_button=True,  #for gradio>6.0:   buttons=["copy"],
            info="Once your run starts, you can view the training progress for your projects here. Click to copy the URL to go to the wandb project and monitor training progress",
        )

        gr.Markdown("### Pre-saved configurations")
        config_saver = ConfigSaver()
        with gr.Row():
            pre_saved_configs = gr.Dropdown(
                label="Pre-saved configurations",
                choices=config_saver.available_configs(),
                value="public-display-4-buttons",
                interactive=True,
                visible=True,
            )

        md1 = gr.Markdown("### Biomechanical Model Parameters")
        bm_params = BMParameters.get_parameters()
        bm_model, ctrl_dt, reset_type, sigdepnoise_enabled, constantnoise_enabled, fatigue_enabled, mj_impl_type = bm_params

        md2 = gr.Markdown("### Task Parameters")
        md21 = gr.Markdown("#### Target Setup")
        num_elements = gr.Number(
            label="Number of Targets",
            value=INIT_ELEMENTS,
            minimum=0,
            maximum=10,
            precision=0,
            interactive=True,
        )

        # Create 10 sets of elements (maximum possible), first INIT_ELEMENTS visible by default
        dynamic_rows = []
        radios = []
        dwell_durations = []
        box_rows = []
        sphere_rows = []

        # Store all components for easy access
        all_components = {"boxes": [], "spheres": []}

        # Create variable that sums up all selected dwell times
        total_dwell_duration = gr.Number(
            label=f"total Dwell Time",
            value=0,
            minimum=0,
            step=0.01,
            interactive=False,
            visible=False,
        )

        for i in range(10):
            # Main row with radio selection
            with gr.Row(visible=(i < INIT_ELEMENTS)) as main_row:
                with gr.Column():
                    radio = gr.Radio(
                        choices=["Box", "Sphere"],
                        label=f"Target {i+1} Type",
                        value="Sphere",
                        interactive=True,
                    )
                    box_row, box_params = BoxParameters.get_parameters(i)
                    (
                        box_position_x,
                        box_position_y,
                        box_position_z,
                        box_size_x_slider,
                        box_size_y_slider,
                        min_touch_force,
                        orientation_angle,
                        rgb_btn,
                    ) = box_params
                    # Store box components
                    all_components["boxes"].append(
                        {
                            key: value
                            for key, value in zip(BoxParameters.fields(), box_params)
                        }
                    )

                    sphere_row, sphere_params = SphereParameters.get_parameters(
                        i, dwell_duration_min=ctrl_dt.value
                    )
                    (
                        x_slider,
                        y_slider,
                        z_slider,
                        size_slider,
                        dwell_duration,
                        color_picker,
                    ) = sphere_params
                    ctrl_dt.change(
                        fn=update_dwell_duration,
                        inputs=(dwell_duration, ctrl_dt),
                        outputs=dwell_duration,
                        preprocess=False,
                    )
                    # Store sphere components
                    all_components["spheres"].append(
                        {
                            key: value
                            for key, value in zip(
                                SphereParameters.fields(), sphere_params
                            )
                        }
                    )

            # Store references
            dynamic_rows.append(main_row)
            radios.append(radio)
            dwell_durations.append(dwell_duration)
            box_rows.append(box_row)
            sphere_rows.append(sphere_row)

        md22 = gr.Markdown("#### Other Task Parameters")
        task_params = TaskParameters.get_parameters(ctrl_dt=ctrl_dt.value)
        (max_duration,) = task_params

        md23 = gr.Markdown("#### Observation Space")
        obs_keys, fatigue_obs_keys, vision_keys, omni_keys = ObservationSpace.get_parameters()

        md24 = gr.Markdown("#### Reward Weights")

        def get_max_dist(num_elements, *radios_and_box_and_sphere_positions):
            ee_pos0 = [0.0, -0.27, 0.37]  # TODO: infer from model!

            _num_targets_max = int(
                len(radios_and_box_and_sphere_positions) // (1 + 2 * 3)
            )
            radios = radios_and_box_and_sphere_positions[:num_elements]
            box_positions_x = radios_and_box_and_sphere_positions[
                _num_targets_max : 2 * _num_targets_max
            ]
            box_positions_y = radios_and_box_and_sphere_positions[
                2 * _num_targets_max : 3 * _num_targets_max
            ]
            box_positions_z = radios_and_box_and_sphere_positions[
                3 * _num_targets_max : 4 * _num_targets_max
            ]
            sphere_positions_x = radios_and_box_and_sphere_positions[
                4 * _num_targets_max : 5 * _num_targets_max
            ]
            sphere_positions_y = radios_and_box_and_sphere_positions[
                5 * _num_targets_max : 6 * _num_targets_max
            ]
            sphere_positions_z = radios_and_box_and_sphere_positions[
                6 * _num_targets_max : 7 * _num_targets_max
            ]
            target_positions = [ee_pos0]
            for target_id in range(num_elements):
                if radios[target_id] == "Box":
                    box_x = box_positions_x[target_id]
                    box_y = box_positions_y[target_id]
                    box_z = box_positions_z[target_id]
                    sanity_check = sanity_check_float_array(
                        box_x, f"Box {target_id+1} Range Position (Depth)"
                    )
                    if not sanity_check:
                        return False, 0
                    sanity_check = sanity_check_float_array(
                        box_y, f"Box {target_id+1} Range Position (Horizontal)"
                    )
                    if not sanity_check:
                        return False, 0
                    sanity_check = sanity_check_float_array(
                        box_z, f"Box {target_id+1} Range Position (Vertical)"
                    )
                    if not sanity_check:
                        return False, 0
                    target_positions.append(
                        [np.mean(box_x), np.mean(box_y), np.mean(box_z)]
                    )
                elif radios[target_id] == "Sphere":
                    sphere_x = sphere_positions_x[target_id]
                    sphere_y = sphere_positions_y[target_id]
                    sphere_z = sphere_positions_z[target_id]
                    sanity_check = sanity_check_float_array(
                        sphere_x, f"Sphere {target_id+1} Range Position (Depth)"
                    )
                    if not sanity_check:
                        return False, 0
                    sanity_check = sanity_check_float_array(
                        sphere_y, f"Sphere {target_id+1} Range Position (Horizontal)"
                    )
                    if not sanity_check:
                        return False, 0
                    sanity_check = sanity_check_float_array(
                        sphere_z, f"Sphere {target_id+1} Range Position (Vertical)"
                    )
                    if not sanity_check:
                        return False, 0
                    target_positions.append(
                        [np.mean(sphere_x), np.mean(sphere_y), np.mean(sphere_z)]
                    )
                else:
                    raise NotImplementedError()
            target_positions = np.array(target_positions)
            max_dist = np.sum(
                np.linalg.norm(np.diff(target_positions, axis=0), axis=1), axis=0
            ).item()
            return True, max_dist

        def update_max_dist(num_elements, *radios_and_box_and_sphere_positions):
            sanity_check, max_dist = get_max_dist(
                num_elements, *radios_and_box_and_sphere_positions
            )
            if not sanity_check:
                return gr.skip()
            return gr.update(value=max_dist)

        radios_and_box_and_sphere_positions = (
            radios
            + [b["box_position_x"] for b in all_components["boxes"]]
            + [b["box_position_y"] for b in all_components["boxes"]]
            + [b["box_position_z"] for b in all_components["boxes"]]
            + [s["x_range"] for s in all_components["spheres"]]
            + [s["y_range"] for s in all_components["spheres"]]
            + [s["z_range"] for s in all_components["spheres"]]
        )
        max_dist = gr.Number(
            label="Maximum Path Length",
            value=get_max_dist(
                num_elements.value,
                *[r.value for r in radios_and_box_and_sphere_positions],
            )[1],
            interactive=False,
            visible=False,
        )
        for k in radios_and_box_and_sphere_positions + [num_elements]:
            k.change(
                fn=update_max_dist,
                inputs=(num_elements, *radios_and_box_and_sphere_positions),
                outputs=max_dist,
                preprocess=False,
            )

        num_ctrls = 26  # TODO: infer from chosen MuJoCo model

        def reward_fct_view(
            num_elements, distance, phase_bonus, done, neural_effort, max_dist
        ):
            return f"""<span style='font-size: 1em;'>The following **Reward** will be provided at each time step *n*, depending on the current target *i*:
            <div align="center">
            $r_n =$ <span title="min: {-1*distance*max_dist:.4g};
            max: {0}">${-1*distance} \\cdot (\\text{{distance to current target }} i + \\sum_{{j=i+1}}^{{{num_elements}}}\\text{{dist(target }}j\\text{{, target }} j-1\\text{{)}})$</span>

            <span title="min: {0};
            max: {1*phase_bonus}">$+ {1*phase_bonus} \\cdot (\\text{{current target }} i \\text{{ successfully hit for the first time}})$</span>
            
            <span title="min: {0};
            max: {1*done}">$+ {1*done} \\cdot (\\text{{task successfully completed}})$</span>
            
            <span title="min: {-1*neural_effort*num_ctrls};
            max: {0}">$- {1*neural_effort} \\cdot (\\text{{squared control effort costs}})$</span>
            </div></span>"""

        _distance_d, _phase_bonus_d, _done_d, _neural_effort_d = (
            RewardFunction.weights_default_min_max_step["distance"][0],
            RewardFunction.weights_default_min_max_step["subtask_bonus"][0],
            RewardFunction.weights_default_min_max_step["completion_bonus"][0],
            RewardFunction.weights_default_min_max_step["neural_effort"][0],
        )
        reward_function_text = gr.Markdown(
            reward_fct_view(
                num_elements=num_elements.value,
                distance=_distance_d,
                phase_bonus=_phase_bonus_d,
                done=_done_d,
                neural_effort=_neural_effort_d,
                max_dist=max_dist.value,
            ),
            elem_id="reward-function",
            line_breaks=True,
            latex_delimiters=[{"left": "$", "right": "$", "display": False}],
        )

        def update_reward_fct_view(
            num_elements, distance, phase_bonus, done, neural_effort, max_dist
        ):
            return gr.update(
                value=reward_fct_view(
                    num_elements=num_elements,
                    distance=distance,
                    phase_bonus=phase_bonus,
                    done=done,
                    neural_effort=neural_effort,
                    max_dist=max_dist,
                )
            )

        (reward_weights,) = RewardFunction.get_parameters()
        weighted_reward_keys_gr = {k.label: k for k in reward_weights}
        for k in reward_weights + [num_elements, max_dist]:
            k.change(
                update_reward_fct_view,
                [
                    num_elements,
                    weighted_reward_keys_gr["distance"],
                    weighted_reward_keys_gr["subtask_bonus"],
                    weighted_reward_keys_gr["completion_bonus"],
                    weighted_reward_keys_gr["neural_effort"],
                    max_dist,
                ],
                reward_function_text,
            )

        md3 = gr.Markdown("### RL Parameters")
        rl_params_and_mds = RLParameters.get_parameters()
        rl_params = rl_params_and_mds[: RLParameters.num_elements]
        md_rl_notes = rl_params_and_mds[-RLParameters.num_mds :]
        num_timesteps = rl_params[0]
        target_init_seed = rl_params[-1]

        gr.Markdown("### Preview of the environment")
        render_button = gr.Button("Update Preview", variant="primary", size="lg")
        load_training_button = gr.Button(
            "Show Training Policy", variant="secondary", size="lg",
        )
        with gr.Row(visible=False) as env_view_row:
            if use_legacy_rendering:
                env_view_1 = gr.Image(label="Environment View", interactive=False)
                env_view_2 = gr.Image(label="Environment View", interactive=False)
            else:
                env_view = gr.HTML()

        gr.Markdown("### Save current configuration")
        with gr.Row():
            config_name_input = gr.Textbox(
                label="Configuration Name", value="", interactive=True
            )
            save_config_button = gr.Button(
                "Save/Update Configuration", variant="primary", size="lg"
            )
        # Add Run button and output
        gr.Markdown("### Send to Training")
        gr.Markdown("""This sets up the current configuration and sends it to the training pipeline.
                    
                    Make sure to run the next notebook cell to start training with this configuration.""")
        with gr.Row():
            run_button = gr.Button("Setup Training", variant="primary", size="lg")

        output_text = gr.Textbox(
            label="Configuration Output",
            lines=20,
            max_lines=30,
            interactive=False,
            show_copy_button=True,  #for gradio>6.0:   buttons=["copy"],
        )

        def args_to_cfg_overrides(run_name, *args):
            """Print all configuration details"""
            # Extract values from args
            num_targets = args[RLParameters.num_elements]
            radio_start = RLParameters.num_elements + 1
            radio_end = radio_start + 10
            radio_values = args[radio_start:radio_end]

            cfg_overrides = [
                "env=universal",
                "run.using_gradio=True",
                f"wandb.project={project_name}",
                "wandb.entity=biom-rl-ui",
                f"wandb.name={run_name}",
            ]
            all_sanity_checks = True
            sanity_check, bm_overrides = BMParameters.parse_values(args)
            cfg_overrides.extend(bm_overrides)
            sanity_check, task_overrides = TaskParameters.parse_values(args)
            all_sanity_checks &= sanity_check
            cfg_overrides.extend(task_overrides)
            sanity_check, obs_overrides = ObservationSpace.parse_values(args)
            all_sanity_checks &= sanity_check
            cfg_overrides.extend(obs_overrides)
            sanity_check, rl_overrides = RLParameters.parse_values(args)
            all_sanity_checks &= sanity_check
            sanity_check, reward_overrides = RewardFunction.parse_values(args)
            all_sanity_checks &= sanity_check
            cfg_overrides.extend(reward_overrides)
            cfg_overrides.extend(rl_overrides)

            for i in range(int(num_targets)):
                target_type = radio_values[i]
                cfg_overrides.append(
                    f"+env/task_config/targets/target_{i}={target_type.lower()}"
                )

                if target_type == "Box":
                    sanity_check, overrides = BoxParameters.parse_values(i, args)
                    all_sanity_checks &= sanity_check
                    if not sanity_check:
                        return False, []
                    cfg_overrides.extend(overrides)

                else:  # Sphere
                    sanity_check, overrides = SphereParameters.parse_values(i, args)
                    all_sanity_checks &= sanity_check
                    if not sanity_check:
                        return False, []
                    cfg_overrides.extend(overrides)

            return all_sanity_checks, cfg_overrides

        def run_training(run_name, *args):
            if run_name == "":
                gr.Warning("Please enter a name for the configuration.")
                text = "Error: You must enter a name for the configuration."
                return text
            sanity_check, cfg_overrides = args_to_cfg_overrides(run_name, *args)
            if not sanity_check:
                gr.Warning("Invalid configuration.")
                return "Error: Invalid configuration."
            # cfg = load_config_interactive(cfg_overrides, cfg_only=True)
            gr.Info("Go back to the notebook and run the next cell to start training!")
            run_state.cfg_overrides.clear()
            run_state.cfg_overrides.extend(cfg_overrides)
            run_state.wandb_run_name = run_name
            # train(cfg)
            text = "Go to the next notebook in the cell and run the training!\n"
            text += "\n".join(run_state.cfg_overrides)
            return text

        def _render_environment_legacy(run_name, *args):
            import mujoco

            target_init_seed = RLParameters.get_target_init_seed(args)
            next_seed = target_init_seed + 1
            sanity_check, cfg_overrides = args_to_cfg_overrides(run_name, *args)
            if not sanity_check:
                gr.Warning("Invalid configuration.")
                return gr.skip()
            config = load_config_interactive(cfg_overrides)

            xml_file = _get_myosuite_root() / config.env.model_path
            if not xml_file.exists():
                gr.Warning(f"Model XML not found in myosuite assets: {config.env.model_path}")
                return gr.skip()

            spec = _universal_spec_fn(xml_file, config)
            model = spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

            imgs = []
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                for cam in ["head_on_camera", "side_camera"]:
                    renderer.update_scene(data, camera=cam)
                    imgs.append(renderer.render())

            return (
                gr.update(value=next_seed),
                gr.update(visible=True),
                gr.update(value=imgs[0]),
                gr.update(value=imgs[1]),
            )

        def _capture_camera_state():
            """Return {position, look_at} for the first connected viser client, or None."""
            clients = _viser_preview_server.get_clients()
            if not clients:
                return None
            client = next(iter(clients.values()))
            try:
                return {
                    "position": client.camera.position.copy(),
                    "look_at": client.camera.look_at.copy(),
                }
            except Exception:
                return None

        def _stop_current_viewer() -> None:
            """Stop the running viewer thread and properly close the viewer."""
            if _render_state["stop_event"] is not None:
                _render_state["stop_event"].set()
            if _render_state["thread"] is not None:
                _render_state["thread"].join(timeout=5.0)
            if _render_state["viewer"] is not None:
                try:
                    _render_state["viewer"].close()
                except Exception:
                    pass
                _render_state["viewer"] = None
            # Reset both 3D scene geometry and GUI control panel so no old
            # elements bleed into the new viewer instance.
            _viser_preview_server.scene.reset()
            import concurrent.futures as _cf
            try:
                with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                    _ex.submit(_viser_preview_server.gui.reset).result(timeout=5)
            except (RuntimeError, _cf.TimeoutError):
                # RuntimeError: viser raises this when a GuiTabHandle tries to
                # update its already-removed parent GuiTabGroupHandle during the
                # reset() traversal — the reset still completes successfully.
                # TimeoutError: gui.reset() can block indefinitely on a hung
                # viser websocket; abandon it after 5 s to unblock training.
                pass

        def _start_viewer(vec_env, stop_event, camera_state=None, poll=False) -> None:
            """Create and start a viewer thread.

            camera_state: dict with 'position' and 'look_at' arrays captured
            from the connected client before teardown, or None on first render.
            Uses _render_state['policy'] if an OnnxPolicy was loaded for this
            render, otherwise falls back to zero actions.
            poll: use PollingViserViewer (background checkpoint polling + modal
            notifications) instead of CameraRestoringViserViewer.
            """
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            inner = _render_state["policy"] or _ZeroActionInner(vec_env.num_actions, device)
            policy = _StoppablePolicy(inner, stop_event)
            viewer_cls = PollingViserViewer if poll else CameraRestoringViserViewer
            viewer = viewer_cls(
                vec_env,
                policy,
                viser_server=_viser_preview_server,
                post_setup_camera=camera_state,
                checkpoint_manager=_render_state.get("checkpoint_manager"),
            )
            thread = threading.Thread(target=_run_viewer, args=(viewer, stop_event), daemon=True)
            _render_state["viewer"] = viewer
            _render_state["stop_event"] = stop_event
            _render_state["thread"] = thread
            thread.start()

        def render_training_result(log_dir=None) -> str:
            """Render the latest checkpoint from a local training log_dir.

            Designed to be called from the notebook after training completes,
            but also works during training as long as the render env already
            exists (guaranteed by prepare_for_training()).
            Falls back to run_state.log_dir when log_dir is not given.
            Returns the viser preview URL for embedding in an IFrame.
            """
            _log_dir = Path(log_dir or run_state.log_dir)
            if not _log_dir.exists():
                print(f"[render] log_dir not found: {_log_dir}")
                return _viser_preview_url

            # Prefer model_final.onnx; otherwise the highest-numbered checkpoint.
            # The runner saves intermediate checkpoints to log_dir/checkpoints/,
            # so scan both the root and that subdirectory.
            final_onnx = _log_dir / "model_final.onnx"
            scan_dirs = [d for d in [_log_dir, _log_dir / "checkpoints"] if d.is_dir()]
            onnx_files = sorted(
                (f for d in scan_dirs for f in d.glob("*.onnx") if is_onnx_checkpoint_name(f.name)),
                key=lambda f: onnx_checkpoint_sort_key(f.name),
            )
            checkpoint = final_onnx if final_onnx.exists() else (onnx_files[-1] if onnx_files else None)
            if checkpoint is None:
                print(f"[render] No .onnx checkpoint found in {_log_dir}")
                return _viser_preview_url

            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            # Rebuild the gradio env using the current cfg_overrides (same as render_environment).
            config = load_config_interactive(run_state.cfg_overrides)
            new_key = _render_struct_key(config)

            _stop_current_viewer()

            needs_rebuild = _render_state["struct_key"] != new_key or _render_state["env"] is None
            if needs_rebuild and run_state.is_training:
                # Cannot create/destroy envs while training holds the CUDA allocator.
                if _render_state["env"] is None:
                    try:
                        gr.Info(
                            "Render environment not ready. "
                            "Call prepare_for_training() before starting training."
                        )
                    except Exception:
                        print("[render] Render env not ready — call prepare_for_training() first.")
                    return _viser_preview_url
                # Struct changed during training: reuse existing env with current geoms.
                needs_rebuild = False

            if needs_rebuild:
                if _render_state["env"] is not None:
                    try:
                        _render_state["env"].close()
                    except Exception:
                        pass
                    _render_state["env"] = None
                    _render_state["vec_env"] = None

                register_mjlab_myouser_task(config)

                env_cfg = load_env_cfg("myoUserUniversal-v0")
                rl_cfg_local = load_rl_cfg("myoUserUniversal-v0")
                env_cfg.scene.num_envs = 1
                env_cfg.viewer.distance = 2.5
                env_cfg.viewer.elevation = 45.0
                env_cfg.viewer.azimuth = 135.0
                env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
                vec_env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg_local.clip_actions)
                _update_model_geoms(env, config)
                _render_state["env"] = env
                _render_state["vec_env"] = vec_env
                _render_state["struct_key"] = new_key
            else:
                _update_model_geoms(_render_state["env"], config)

            # Load policy via OnnxCheckpointingMjlabRunner — identical to play_notebook_viser.py.
            vec_env = _render_state["vec_env"]
            rl_cfg_local = load_rl_cfg("myoUserUniversal-v0")
            runner = OnnxCheckpointingMjlabRunner(
                vec_env, asdict(rl_cfg_local), str(checkpoint.parent), device,
                task_id="myoUserUniversal-v0",
            )
            runner.load_onnx(checkpoint)
            policy = runner.get_inference_policy(device=device)
            print(f"[render] Loaded checkpoint: {checkpoint.name} from {_log_dir.name}")

            ckpt_dir = checkpoint.parent

            def _reload(path: str):
                runner.load_onnx(Path(path))
                return runner.get_inference_policy(device=device)

            def _fetch_ckpts() -> list:
                now = _time.time()
                entries = []
                for f in ckpt_dir.glob("*.onnx"):
                    if not is_onnx_checkpoint_name(f.name):
                        continue
                    entries.append((f.name, format_time_ago(int(now - f.stat().st_mtime))))
                entries.sort(key=lambda x: onnx_checkpoint_sort_key(x[0]))
                return entries

            ckpt_manager = CheckpointManager(
                current_name=checkpoint.name,
                fetch_available=_fetch_ckpts,
                load_checkpoint=lambda name: _reload(str(ckpt_dir / name)),
            )

            _render_state["policy"] = policy
            _render_state["checkpoint_manager"] = ckpt_manager

            stop_event = threading.Event()
            _start_viewer(vec_env, stop_event, poll=True)

            return _viser_preview_url

        def render_environment(run_name, *args):
            target_init_seed = RLParameters.get_target_init_seed(args)
            next_seed = target_init_seed + 1
            sanity_check, cfg_overrides = args_to_cfg_overrides(run_name, *args)
            if not sanity_check:
                gr.Warning("Invalid configuration.")
                return gr.skip()

            import concurrent.futures as _cf

            def _core():
                config = load_config_interactive(cfg_overrides)
                device = "cuda:0" if torch.cuda.is_available() else "cpu"

                new_key = _render_struct_key(config)
                structural_change = _render_state["struct_key"] != new_key or _render_state["env"] is None

                # Capture camera before teardown so we can restore it afterwards.
                camera_state = _capture_camera_state()
                _stop_current_viewer()

                if structural_change and run_state.is_training:
                    # Cannot create/destroy envs while training holds the CUDA allocator.
                    if _render_state["env"] is None:
                        gr.Warning(
                            "Render environment not ready. "
                            "Call prepare_for_training() before starting training."
                        )
                        return gr.skip()
                    # Config changed during training: reuse existing env, update geoms only.
                    gr.Warning(
                        "Some of the requested changes can only be applied after training. "
                        "The current rendering may not fully reflect the new configuration."
                    )
                    structural_change = False

                if structural_change:
                    # Full restart: rebuild env from new registration.
                    if _render_state["env"] is not None:
                        try:
                            _render_state["env"].close()
                        except Exception:
                            pass
                        _render_state["env"] = None
                        _render_state["vec_env"] = None

                    register_mjlab_myouser_task(config)

                    env_cfg = load_env_cfg("myoUserUniversal-v0")
                    rl_cfg = load_rl_cfg('myoUserUniversal-v0')
                    env_cfg.scene.num_envs = 1
                    # Place the camera closer to the hand workspace (only used when
                    # no prior camera state exists, i.e. the very first render).
                    env_cfg.viewer.distance = 2.5
                    env_cfg.viewer.elevation = 45.0
                    env_cfg.viewer.azimuth = 135.0
                    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
                    vec_env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
                    # Set mj_model colors/sizes to the configured values so the viewer
                    # bakes them correctly.  The spec builds the model with a random DR
                    # sample; _update_model_geoms overwrites that with config averages.
                    _update_model_geoms(env, config)
                    _render_state["env"] = env
                    _render_state["vec_env"] = vec_env
                    _render_state["struct_key"] = new_key
                else:
                    # Fast path: non-structural change (same model, same targets).
                    # Update mj_model colors/sizes (what viser bakes at setup() time)
                    # and event term params (what env.reset() samples for positions).
                    # Restarting the viewer is enough to pick up color/size changes;
                    # no env rebuild needed (~1 s vs ~30 s).
                    _update_model_geoms(_render_state["env"], config)
                    try:
                        _render_state["env"].reset()
                    except Exception:
                        pass

                # Load checkpoint policy directly from the dropdown selections,
                # bypassing config so the values are always current at call time.
                # args[6] = select_checkpoint_run, args[7] = select_checkpoint_file
                # (RLParameters fields 6 and 7 within run_inputs).
                checkpoint_run = args[6]
                checkpoint_file = args[7]
                policy, ckpt_manager = _build_policy_and_ckpt_manager(
                    checkpoint_run, checkpoint_file, device
                )
                _render_state["policy"] = policy
                _render_state["checkpoint_manager"] = ckpt_manager

                stop_event = threading.Event()
                _start_viewer(_render_state["vec_env"], stop_event, camera_state=camera_state)

                iframe_html = (
                    f'<iframe src="{_viser_preview_url}" '
                    f'width="100%" height="520px" frameborder="0" '
                    f'allow="cross-origin-isolated"></iframe>'
                )
                return (
                    gr.update(value=next_seed),
                    gr.update(visible=True),
                    gr.update(value=iframe_html),
                )

            try:
                with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                    return _ex.submit(_core).result(timeout=90)
            except _cf.TimeoutError:
                gr.Warning("Preview update timed out after 90 s — please try again.")
                return gr.skip()

        def update_dynamic_elements(num):
            """Show/hide dynamic rows based on the number input"""
            num = max(0, min(int(num) if num is not None else 0, 10))
            updates = []
            for i in range(10):
                if i < num:
                    updates.append(gr.update(visible=True))
                else:
                    updates.append(gr.update(visible=False))
            return updates

        def update_fatigue_obs_keys_interactive(fatigue_enabled, *fatigue_obs_keys):
            """Update interactability of fatigue observation keys based on fatigue_enabled"""
            updates = []
            for k in fatigue_obs_keys:
                if fatigue_enabled:
                    updates.append(gr.update(interactive=True))
                else:
                    updates.append(gr.update(value=False, interactive=False))
            return updates

        def update_omni_key_presets(vision_key, *omni_keys):
            """Update/Override default values of omni keys if vision_key is enabled"""
            updates = []
            for k in omni_keys:
                updates.append(gr.update(value=False) if vision_key else gr.update())
            return updates

        def update_num_timesteps(bm_model, num_elements, ctrl_dt, fatigue_enabled, *dwell_durations_and_radios):
            _num_targets_max = int(len(dwell_durations_and_radios) // 2)
            dwell_durations = dwell_durations_and_radios[:num_elements]
            radios = dwell_durations_and_radios[
                _num_targets_max : _num_targets_max + num_elements
            ]
            total_dwell_duration = sum(
                map(
                    lambda x, y: max(0, x - ctrl_dt) * (y == "Sphere"),
                    dwell_durations,
                    radios,
                )
            )

            target_value = (
                15_000_000
                + (bm_model in ("MoBL_Arms_Hand", "MyoArm")) * 3_000_000   #number to be validated
                + (bm_model in ("MyoArm_nohand", "MyoArm")) * 10_000_000   #number to be validated
                + (fatigue_enabled) * 20_000_000  #number to be validated
                + max((num_elements - 3), 0) * 1_000_000
                + int(total_dwell_duration // 0.3) * 1_000_000
            )
            return gr.update(value=target_value)

        def toggle_interface_type(radio_value):
            """Show appropriate interface based on radio selection"""
            if radio_value == "Box":
                return gr.update(visible=True), gr.update(visible=False)
            else:  # Sphere
                return gr.update(visible=False), gr.update(visible=True)

        def update_pageview(pageview, *advanced_options):
            visible = pageview == "Advanced"
            return [gr.update(visible=visible) for opt in advanced_options]

        advanced_options_and_markdowns = [
            *bm_params,
            *task_params,
            *obs_keys,
            *fatigue_obs_keys,
            *vision_keys,
            *omni_keys,
            *rl_params,
            *md_rl_notes,
        ] + [md1, md22, md23, md3]
        pageview.change(
            fn=update_pageview,
            inputs=(pageview, *advanced_options_and_markdowns),
            outputs=advanced_options_and_markdowns,
        )
        # also call this method once at the very beginning
        demo.load(
            fn=update_pageview,
            inputs=(pageview, *advanced_options_and_markdowns),
            outputs=advanced_options_and_markdowns,
        )

        # Event handler for dynamic elements
        num_elements.change(
            update_dynamic_elements, inputs=num_elements, outputs=dynamic_rows
        )
        
        # Event handler for fatigue boolean to fatigue obs keys
        fatigue_enabled.change(
            update_fatigue_obs_keys_interactive,
            inputs=(fatigue_enabled, *fatigue_obs_keys),
            outputs=fatigue_obs_keys,
            preprocess=False,
        )

        # Event handler for vision checkbox to omni keys
        for vision_key in vision_keys:
            vision_key.change(update_omni_key_presets,
                              inputs=(vision_key, *omni_keys),
                              outputs=omni_keys,
                              preprocess=False)

        # Event handler for biomechanical model to suggested number of training steps
        bm_model.change(
            update_num_timesteps,
            inputs=(bm_model, num_elements, ctrl_dt, fatigue_enabled, *dwell_durations, *radios),
            outputs=num_timesteps,
            preprocess=False,
        )

        # Event handler for fatigue (enabled/disabled) to suggested number of training steps
        fatigue_enabled.change(
            update_num_timesteps,
            inputs=(bm_model, num_elements, ctrl_dt, fatigue_enabled, *dwell_durations, *radios),
            outputs=num_timesteps,
            preprocess=False,
        )

        # Event handler for number of targets to num of training steps
        num_elements.change(
            update_num_timesteps,
            inputs=(bm_model, num_elements, ctrl_dt, fatigue_enabled, *dwell_durations, *radios),
            outputs=num_timesteps,
            preprocess=False,
        )

        # Event handlers for each radio button to control interface type and num of training steps
        for i in range(10):
            radios[i].change(
                toggle_interface_type,
                inputs=radios[i],
                outputs=[box_rows[i], sphere_rows[i]],
            )
            radios[i].change(
                update_num_timesteps,
                inputs=(bm_model, num_elements, ctrl_dt, fatigue_enabled, *dwell_durations, *radios),
                outputs=num_timesteps,
                preprocess=False,
            )

        # Event handler for each dwell duration to num of training steps
        for i in range(10):
            dwell_durations[i].change(
                update_num_timesteps,
                inputs=(bm_model, num_elements, ctrl_dt, fatigue_enabled, *dwell_durations, *radios),
                outputs=num_timesteps,
                preprocess=False,
            )

        # Prepare inputs for run button
        run_inputs = [*rl_params, num_elements]
        run_inputs.extend(radios)

        # Add all box components
        for i in range(10):
            for key in BoxParameters.fields():
                run_inputs.append(all_components["boxes"][i][key])

        # Add all sphere components
        for i in range(10):
            for key in SphereParameters.fields():
                run_inputs.append(all_components["spheres"][i][key])

        run_inputs.extend(bm_params)
        run_inputs.extend(task_params)
        run_inputs.extend(obs_keys)
        run_inputs.extend(fatigue_obs_keys)
        run_inputs.extend(vision_keys)
        run_inputs.extend(omni_keys)
        run_inputs.extend(reward_weights)

        # Run button event
        run_or_save_inputs = [config_name_input] + run_inputs
        run_button.click(run_training, inputs=run_or_save_inputs, outputs=output_text)

        config_saver.run_inputs = run_inputs

        pre_saved_configs.change(
            config_saver.load_config, inputs=pre_saved_configs, outputs=run_inputs
        )
        # Event handler for suggested config name (also load once at startup)
        _update_suggest_config_name = lambda config_name: gr.update(value=config_name) if config_name else gr.update()
        pre_saved_configs.change(
            _update_suggest_config_name,
            inputs=pre_saved_configs,
            outputs=config_name_input,
        )
        demo.load(_update_suggest_config_name, inputs=pre_saved_configs, outputs=config_name_input)

        save_config_button.click(
            config_saver.config_save_clicked,
            inputs=run_or_save_inputs,
            outputs=pre_saved_configs,
        )

        if use_legacy_rendering:
            render_button.click(
                _render_environment_legacy,
                inputs=run_or_save_inputs,
                outputs=[target_init_seed, env_view_row, env_view_1, env_view_2],
            )
            # Load saved config then render on startup.
            demo.load(config_saver.load_config, inputs=pre_saved_configs, outputs=run_inputs)
        else:
            render_button.click(
                render_environment,
                inputs=run_or_save_inputs,
                outputs=[target_init_seed, env_view_row, env_view],
            )

            def _load_training_handler():
                if not run_state.log_dir:
                    gr.Warning("No training run found. Run the training cell first.")
                    return gr.skip(), gr.skip()
                url = render_training_result()
                iframe_html = (
                    f'<iframe src="{url}" width="100%" height="520px" '
                    f'frameborder="0" allow="cross-origin-isolated"></iframe>'
                )
                return gr.update(visible=True), gr.update(value=iframe_html)

            load_training_button.click(
                _load_training_handler,
                inputs=[],
                outputs=[env_view_row, env_view],
            )

            # Load saved config first, then immediately render with the updated values.
            demo.load(
                config_saver.load_config,
                inputs=pre_saved_configs,
                outputs=run_inputs,
            ).then(
                render_environment,
                inputs=run_or_save_inputs,
                outputs=[target_init_seed, env_view_row, env_view],
            )

    # Expose helpers so the notebook can trigger rendering after training.
    run_state.render_training_result = render_training_result
    run_state._viser_preview_url = _viser_preview_url

    def _prepare_for_training() -> None:
        """Prepare the gradio side for training.

        - Stops the viser viewer thread (no CUDA allocation/deallocation during training).
        - Ensures the render env is initialised now while CUDA is quiet, so the GUI
          buttons (Update Preview, Show Training Policy) remain functional at any point
          during training without needing to create a new env under load.
        - Calls torch.cuda.empty_cache() to defragment the allocator before training starts.

        IMPORTANT: the render env is intentionally kept alive so that viewer restarts
        during training only update the policy/checkpoint — they never touch env creation.
        """
        _stop_current_viewer()

        # Ensure the render env exists and matches the current cfg_overrides before training
        # starts (while CUDA is quiet).  If cfg changed since the last render, close the stale
        # env and rebuild so struct_key is consistent throughout training.
        if run_state.cfg_overrides:
            try:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                config = load_config_interactive(run_state.cfg_overrides)
                new_key = _render_struct_key(config)
                if _render_state["env"] is not None and _render_state["struct_key"] != new_key:
                    try:
                        _render_state["env"].close()
                    except Exception:
                        pass
                    _render_state["env"] = None
                    _render_state["vec_env"] = None
                    print("[prepare] Config changed — rebuilding render env for training.")
                if _render_state["env"] is None:
                    register_mjlab_myouser_task(config)
                    env_cfg = load_env_cfg("myoUserUniversal-v0")
                    rl_cfg_local = load_rl_cfg("myoUserUniversal-v0")
                    env_cfg.scene.num_envs = 1
                    env_cfg.viewer.distance = 2.5
                    env_cfg.viewer.elevation = 45.0
                    env_cfg.viewer.azimuth = 135.0
                    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
                    vec_env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg_local.clip_actions)
                    _update_model_geoms(env, config)
                    _render_state["env"] = env
                    _render_state["vec_env"] = vec_env
                    _render_state["struct_key"] = new_key
                    print("[prepare] Render env initialised for training.")
                else:
                    print("[prepare] Render env already up-to-date for training.")
            except Exception as exc:
                print(f"[prepare] Could not pre-create render env: {exc}")

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    run_state.prepare_for_training = _prepare_for_training

    return demo


def launch_training(state):
    # Stop any running viewer and release the render env so the training process
    # gets a clean, fully-defragmented CUDA allocator.  This is what prevents the
    # RuntimeError on first launch that previously required a retry.
    if hasattr(state, 'prepare_for_training'):
        state.prepare_for_training()

    config = load_config_interactive(state.cfg_overrides)
    register_mjlab_myouser_task(config)

    configure_torch_backends()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        print("[WARNING] CUDA is not available. Running on CPU may be very slow for training.")

    env_cfg = load_env_cfg('myoUserUniversal-v0')
    rl_cfg = load_rl_cfg('myoUserUniversal-v0')

    resume_path = None
    if config.rl.load_checkpoint_path is not None and config.rl.load_checkpoint_path != "None":
        # OnnxCheckpointingMjlabRunner saves .onnx; translate the default
        # 'model_.*.pt' regex (and any explicit .pt name) before scanning.
        resume_path = config.rl.load_checkpoint_path.replace(".pt", ".onnx")
        print(f"[INFO] Resuming from checkpoint: {resume_path}")

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = Path(LOG_DIR) / f'{timestamp}_{state.wandb_run_name}'
    log_dir.mkdir(parents=True, exist_ok=True)
    state.log_dir = str(log_dir)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = OnnxCheckpointingMjlabRunner(
        vec_env, asdict(rl_cfg), str(log_dir), device, task_id='myoUserUniversal-v0'
    )
    state.is_training = True
    try:
        if resume_path is not None:
            runner.load(str(resume_path))  # delegates to load_onnx() for .onnx files
        runner.learn(num_learning_iterations=rl_cfg.max_iterations, init_at_random_ep_len=True)
        runner.save(str(log_dir / 'model_final.pt'))
    finally:
        state.is_training = False
        env.close()


if __name__ == "__main__":
    wandb_url = None
    demo = get_ui(wandb_url)
    demo.launch(share=True)
