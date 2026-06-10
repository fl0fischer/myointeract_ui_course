"""Hydra CLI machinery for myouser configs.

All dataclass definitions live in myouser_configs.py (pure Python, no Hydra
dependency).  This module adds the OmegaConf resolver registration, the
ConfigStore setup, and the interactive helper used by notebooks / scripts.
"""
import math

import hydra
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from ml_collections import ConfigDict
from omegaconf import OmegaConf, open_dict

from myosuite.core.myouser_configs import (
    BaseEnvConfig,
    ButtonTarget,
    Config,
    LIST_CONFIGS,
    NetworkConfig,
    PointingEnvConfig,
    PointingTarget,
    RLConfig,
    RunConfig,
    TrackingEnvConfig,
    UniversalConfig,
    UniversalEnvConfig,
    VisionDisabledConfig,
    VisionEnabledConfig,
    VisionNetworkConfig,
    WANDBDisabledConfig,
    WANDBEnabledConfig,
)

# Re-export so callers that did `from myouser_hydra_cli import XConfig` still work.
__all__ = [
    "BaseEnvConfig", "ButtonTarget", "Config", "LIST_CONFIGS",
    "NetworkConfig", "PointingEnvConfig", "PointingTarget", "RLConfig",
    "RunConfig", "TrackingEnvConfig", "UniversalConfig", "UniversalEnvConfig",
    "VisionDisabledConfig", "VisionEnabledConfig", "VisionNetworkConfig",
    "WANDBDisabledConfig", "WANDBEnabledConfig",
    "load_config_interactive", "gradio_json_to_overrides",
]


# ---------------------------------------------------------------------------
# OmegaConf resolvers
# ---------------------------------------------------------------------------

def _select_network(vision_enabled: str) -> str:
    return "vision" if vision_enabled == "enabled" else "no_vision"


def _select_targets(num_targets: int):
    from myosuite.core.myouser_configs import (
        EightTargetConfig, FiveTargetConfig, FourTargetConfig,
        NineTargetConfig, OneTargetConfig, SevenTargetConfig,
        SixTargetConfig, TenTargetConfig, ThreeTargetConfig, TwoTargetConfig,
    )
    configs = [
        None,
        OneTargetConfig(), TwoTargetConfig(), ThreeTargetConfig(),
        FourTargetConfig(), FiveTargetConfig(), SixTargetConfig(),
        SevenTargetConfig(), EightTargetConfig(), NineTargetConfig(),
        TenTargetConfig(),
    ]
    if 1 <= num_targets <= 10:
        return configs[num_targets]
    raise ValueError(f"num_targets must be between 1 and 10, got {num_targets}")


OmegaConf.register_new_resolver("check_string", lambda x: "" if x is None else "-" + str(x))
OmegaConf.register_new_resolver("select_network", _select_network)
OmegaConf.register_new_resolver("select_targets", _select_targets)
OmegaConf.register_new_resolver("int_divide", lambda x, y: int(float(x) // float(y)))


# ---------------------------------------------------------------------------
# ConfigStore registration
# ---------------------------------------------------------------------------

def register_universal_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_universal", node=UniversalConfig)
    for i, name, config in LIST_CONFIGS:
        cs.store(group="env/task_config/targets", name=name, node=config)
        for j in range(i + 1):
            cs.store(group=f"env/task_config/targets/target_{j}", name="sphere", node=PointingTarget)
            cs.store(group=f"env/task_config/targets/target_{j}", name="box",    node=ButtonTarget)
    cs.store(name="config", node=Config)
    cs.store(group="wandb",             name="enabled",      node=WANDBEnabledConfig)
    cs.store(group="wandb",             name="disabled",     node=WANDBDisabledConfig)
    cs.store(group="vision",            name="enabled",      node=VisionEnabledConfig)
    cs.store(group="vision",            name="disabled",     node=VisionDisabledConfig)
    cs.store(group="env",               name="base_env_config", node=BaseEnvConfig)
    cs.store(group="env",               name="pointing",     node=PointingEnvConfig)
    cs.store(group="env",               name="tracking",     node=TrackingEnvConfig)
    cs.store(group="env",               name="universal",    node=UniversalEnvConfig)
    cs.store(group="rl",                name="rl_config",    node=RLConfig)
    cs.store(group="run",               name="run",          node=RunConfig)
    cs.store(group="rl/network_factory", name="vision",      node=VisionNetworkConfig)
    cs.store(group="rl/network_factory", name="no_vision",   node=NetworkConfig)


# ---------------------------------------------------------------------------
# Interactive config loader (used from notebooks and training scripts)
# ---------------------------------------------------------------------------

def load_config_interactive(overrides: list[str] = (), cfg_only: bool = False):
    """Compose a myouser config from a list of Hydra override strings.

    Example::

        config = load_config_interactive([
            "env=universal",
            "env/task_config/targets=four",
            "env.ctrl_dt=0.05",
        ])

    Returns a ``ml_collections.ConfigDict`` (or the raw OmegaConf DictConfig
    when *cfg_only* is True).
    """
    register_universal_configs()
    GlobalHydra.instance().clear()
    config_name = "config_universal" if "env=universal" in overrides else "config"
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    if cfg_only:
        return cfg
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    container["env"]["vision"] = container["vision"]
    return ConfigDict(container)


# ---------------------------------------------------------------------------
# gradio JSON → Hydra overrides converter
# ---------------------------------------------------------------------------

_POSSIBLE_OBS_KEYS     = ["qpos", "qvel", "qacc", "ee_pos", "act"]
_POSSIBLE_FATIGUE_KEYS = ["MA", "MR", "MF"]
_POSSIBLE_VISION_KEYS  = ["rgb", "depth"]
_POSSIBLE_OMNI_KEYS    = ["target_pos", "target_size", "phase_progress", "dwell_fraction"]
_NUM_TARGETS_TEXT      = ["", "one", "two", "three", "four", "five",
                          "six", "seven", "eight", "nine", "ten"]
_BM_MODEL_MAP = {
    "MoBL_Arms_Index": (
        "myosuite/envs/myo/assets/arm/mobl_arms_index_universal_myouser.xml",
        "fingertip", "humphant",
    ),
    "MoBL_Arms_Hand": (
        "myosuite/envs/myo/assets/arm/mobl_arms_hand_universal_myouser.xml",
        "IFtip", "humphant",
    ),
    "MyoArm_nohand": (
        "myosuite/envs/myo/assets/arm/myoarm_nohand_universal_myouser.xml",
        "IFtip", "R.Shoulder_marker",
    ),
    "MyoArm": (
        "myosuite/envs/myo/assets/arm/myoarm_universal_myouser.xml",
        "IFtip", "R.Shoulder_marker",
    ),
}


def _parse_rgba(rgba_str: str) -> list[float]:
    """'rgba(R,G,B,1)' or '#rrggbb' → [r, g, b] normalised to 0–1."""
    if "rgba(" in rgba_str:
        inner = rgba_str.split("rgba(")[1].rstrip(")")
        r, g, b, *_ = (float(x.strip()) for x in inner.split(","))
        return [r / 255.0, g / 255.0, b / 255.0]
    hex_color = rgba_str.lstrip("#")
    return [int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def gradio_json_to_overrides(cfg: dict) -> list[str]:
    """Convert a saved gradio JSON config dict to Hydra override strings.

    The JSON format is produced by ``ConfigSaver.to_labelled_dict`` in
    ``gradio_ui.py``.  This function mirrors the logic of
    ``args_to_cfg_overrides`` without any dependency on gradio or the
    form-value flat-list encoding.

    The returned list can be passed directly to ``load_config_interactive``.
    """
    overrides: list[str] = ["env=universal", "run.using_gradio=True"]

    # ── RL parameters ────────────────────────────────────────────────────────
    rl = cfg.get("rl", {})
    num_targets = int(cfg.get("num_targets", 1))
    overrides += [
        f"env/task_config/targets={_NUM_TARGETS_TEXT[num_targets]}",
        f"rl.num_timesteps={int(rl.get('Number of Training Steps', 1_000_000))}",
        f"rl.save_interval={int(rl.get('Checkpoint Interval (in Number of Training Iterations)', 30))}",
        f"rl.num_envs={int(rl.get('Number of Parallel Environments', 4096))}",
        f"rl.unroll_length={int(rl.get('Unroll Length', 10))}",
        f"rl.num_minibatches={int(rl.get('Number of Minibatches', 8))}",
        f"rl.num_epochs_per_update={int(rl.get('Number of Epochs Per Update', 8))}",
        f"env.task_config.target_init_seed={int(rl.get('Target Initial Seed', 0))}",
    ]

    # ── Biomechanical model ───────────────────────────────────────────────────
    bm = cfg.get("bm", {})
    bm_model = bm.get("bm_model", "MoBL_Arms_Index")
    model_path, ee_site, ref_site = _BM_MODEL_MAP[bm_model]
    sigdepnoise = "white" if bm.get("sigdepnoise_enabled", False) else "None"
    constnoise  = "white" if bm.get("constantnoise_enabled", False) else "None"
    overrides += [
        f"env.ctrl_dt={float(bm.get('ctrl_dt', 0.05))}",
        f"env.task_config.reset_type={bm.get('reset_type', 'epsilon_uniform')}",
        f"env.muscle_config.noise_params.sigdepnoise_type={sigdepnoise}",
        f"env.muscle_config.noise_params.constantnoise_type={constnoise}",
        f"env.muscle_config.fatigue_enabled={bm.get('fatigue_enabled', False)}",
        f"env.mj_impl={bm.get('mj_impl_type', 'warp')}",
        f"env.model_path={model_path}",
        f"env.task_config.reach_settings.ref_site={ref_site}",
        f"env.task_config.reach_settings.ee_site={ee_site}",
    ]

    # ── Task parameters ───────────────────────────────────────────────────────
    task = cfg.get("task", {})
    overrides.append(f"env.task_config.max_duration={float(task.get('max_duration', 4.0))}")

    # ── Observation space ─────────────────────────────────────────────────────
    # JSON: {'<field_0>': [bool, ...], '<field_1>': [...], ...} where the values
    # are [obs_keys bools, fatigue_keys bools, vision_keys bools, omni_keys bools]
    # stored in insertion order (the key names themselves are irrelevant artifacts
    # of ConfigSaver.to_labelled_dict's zip-based encoding).
    obs_groups   = list(cfg.get("obs", {}).values())
    obs_bools    = obs_groups[0] if len(obs_groups) > 0 else [True] * len(_POSSIBLE_OBS_KEYS)
    fatigue_bools = obs_groups[1] if len(obs_groups) > 1 else []
    vision_bools  = obs_groups[2] if len(obs_groups) > 2 else []
    omni_bools    = obs_groups[3] if len(obs_groups) > 3 else []

    obs_selected     = [k for k, v in zip(_POSSIBLE_OBS_KEYS,     obs_bools)    if v]
    fatigue_selected = [k for k, v in zip(_POSSIBLE_FATIGUE_KEYS, fatigue_bools) if v]
    omni_selected    = [k for k, v in zip(_POSSIBLE_OMNI_KEYS,    omni_bools)   if v]
    vision_selected  = [k for k, v in zip(_POSSIBLE_VISION_KEYS,  vision_bools) if v]
    vision_enabled   = bool(vision_selected)
    overrides += [
        f"env.task_config.obs_keys={obs_selected}",
        f"env.muscle_config.fatigue_obs_keys={fatigue_selected}",
        f"vision={'enabled' if vision_enabled else 'disabled'}",
        f"env.task_config.omni_keys={omni_selected}",
    ]
    if vision_enabled:
        if "rgb" in vision_selected and "depth" in vision_selected:
            mode = "rgbd"
        elif "rgb" in vision_selected:
            mode = "rgb"
        else:
            mode = "depth"
        overrides.append(f"vision.vision_mode={mode}")

    # ── Reward weights ────────────────────────────────────────────────────────
    _rename = {"subtask_bonus": "phase_bonus", "completion_bonus": "done"}
    reward = cfg.get("reward", {})
    _reward_defaults: dict = {}
    for key in ["distance", "subtask_bonus", "completion_bonus", "neural_effort", "jac_effort"]:
        hydra_key = _rename.get(key, key)
        overrides.append(
            f"env.task_config.weighted_reward_keys.{hydra_key}={reward.get(key, _reward_defaults.get(key, 0))}"
        )

    # ── Per-target geometry ───────────────────────────────────────────────────
    for i in range(num_targets):
        t = cfg.get(f"target_{i + 1}", {})
        target_type = t.get("radio", "Sphere")
        overrides.append(f"+env/task_config/targets/target_{i}={target_type.lower()}")

        if target_type == "Box":
            b = t.get("box", {})
            px = b.get("box_position_x", [0.3, 0.3])
            py = b.get("box_position_y", [0.0, 0.0])
            pz = b.get("box_position_z", [0.0, 0.0])
            # Older configs stored scalars; newer ones store [min, max] pairs.
            if not isinstance(px, list): px = [px, px]
            if not isinstance(py, list): py = [py, py]
            if not isinstance(pz, list): pz = [pz, pz]
            sx = b.get("box_size_x", [0.05, 0.05])
            sy = b.get("box_size_y", [0.05, 0.05])
            if not isinstance(sx, list): sx = [sx, sx]
            if not isinstance(sy, list): sy = [sy, sy]
            angle = float(b.get("orientation_angle", 0))
            rgb   = _parse_rgba(b.get("rgb", "rgba(204.0,25.5,25.5,1)"))
            overrides += [
                f"env.task_config.targets.target_{i}.position="
                f"[[{px[0]},{py[0]},{pz[0]}],[{px[1]},{py[1]},{pz[1]}]]",
                f"env.task_config.targets.target_{i}.size="
                f"[[{0.5*sx[0]},{0.5*sy[0]},0.01],[{0.5*sx[1]},{0.5*sy[1]},0.01]]",
                f"env.task_config.targets.target_{i}.min_touch_force={b.get('min_touch_force', 1.0)}",
                f"env.task_config.targets.target_{i}.euler=[0,{-angle * math.pi / 180},0]",
                f"env.task_config.targets.target_{i}.rgb=[{rgb[0]},{rgb[1]},{rgb[2]}]",
            ]
        else:  # Sphere
            s   = t.get("sphere", {})
            xr  = s.get("x_range",    [0.3, 0.3])
            yr  = s.get("y_range",    [-0.1, 0.1])
            zr  = s.get("z_range",    [-0.3, 0.3])
            sr  = s.get("size_range", [0.05, 0.15])
            rgb = _parse_rgba(s.get("rgb", "rgba(204.0,25.5,25.5,1)"))
            overrides += [
                f"env.task_config.targets.target_{i}.position="
                f"[[{xr[0]},{yr[0]},{zr[0]}],[{xr[1]},{yr[1]},{zr[1]}]]",
                f"env.task_config.targets.target_{i}.size=[{sr[0]},{sr[1]}]",
                f"env.task_config.targets.target_{i}.dwell_duration={s.get('dwell_duration', 0.25)}",
                f"env.task_config.targets.target_{i}.rgb=[{rgb[0]},{rgb[1]},{rgb[2]}]",
            ]

    return overrides


# ---------------------------------------------------------------------------
# Hydra entry-point (for direct CLI use)
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_name="config")
def my_app(cfg: Config) -> None:
    if getattr(cfg.env, "env_name", None) == "MyoUserUniversal":
        with open_dict(cfg):
            cfg.env.task_config.targets = "default"
    container = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    print(OmegaConf.to_yaml(container))


if __name__ == "__main__":
    register_universal_configs()
    my_app()
