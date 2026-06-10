from dataclasses import dataclass, field, make_dataclass
from enum import Enum
from typing import Any, Dict, List, Type, Union
from omegaconf import MISSING, OmegaConf

@dataclass
class NoiseParams:
    sigdepnoise_type: Union[str, None] = None
    sigdepnoise_level: float = 0.103
    constantnoise_type: Union[str, None] = None
    constantnoise_level: float = 0.185

@dataclass
class MuscleConfig:
    muscle_condition: Union[str, None] = None  #TODO: remove as deprecated
    sex: Union[str, None] = None
    control_type: str = "default"
    fatigue_enabled: bool = False
    fatigue_persist_across_episodes: bool = False
    fatigue_reset_vec: Union[List[float], None] = None
    fatigue_reset_random: bool = False
    fatigue_obs_keys: List[str] = field(default_factory=lambda: [])
    noise_params: NoiseParams = field(default_factory=lambda: NoiseParams())
    dep_enabled: bool = False  #TODO: move into RL config? (requires passing ppo_params or even entire config to registry.load()...)
    dep_stochastic_switch: bool = True

@dataclass
class BaseEnvConfig:
    env_name: str = MISSING
    model_path: str = MISSING
    mj_impl: str = "mjlab"
    ctrl_dt: float = 0.002 * 25
    sim_dt: float = 0.002
    muscle_config: MuscleConfig = field(default_factory=lambda: MuscleConfig())
    eval_mode: bool = False

# OmegaConf.register_new_resolver(
#     "check_string", lambda x: "" if x is None else "-" + str(x)
# )


def select_network(vision_enabled):
    if vision_enabled == "enabled":
        return "vision"
    else:
        return "no_vision"


def select_targets(num_targets):
    target_configs = [
        None,
        OneTargetConfig(),
        TwoTargetConfig(),
        ThreeTargetConfig(),
        FourTargetConfig(),
        FiveTargetConfig(),
        SixTargetConfig(),
        SevenTargetConfig(),
        EightTargetConfig(),
        NineTargetConfig(),
        TenTargetConfig(),
    ]

    if 1 <= num_targets <= 10:
        return target_configs[num_targets]
    else:
        raise ValueError(f"num_targets must be between 1 and 10, got {num_targets}")


# OmegaConf.register_new_resolver("select_network", select_network)
# OmegaConf.register_new_resolver("select_targets", select_targets)

@dataclass
class WANDBEnabledConfig:
    enabled: bool = True
    entity: Union[str, None] = (
        "biom-rl-ui"  # Set to this by default, choose a different entity for personal projects
    )
    name: Union[str, None] = (
        "${env.env_name}-${now:%Y%m%d}-${now:%H%M%S}${check_string:${run.suffix}}"
    )
    project: str = "MJXRL"
    tags: Union[List[str], None] = None
    group: Union[str, None] = None


@dataclass
class WANDBDisabledConfig:
    enabled: bool = False
    entity: Union[str, None] = (
        "biom-rl-ui"  # Set to this by default, choose a different entity for personal projects
    )
    name: Union[str, None] = (
        "${env.env_name}-${now:%Y%m%d}-${now:%H%M%S}${check_string:${run.suffix}}"
    )
    project: str = "MJXRL"
    tags: Union[List[str], None] = None
    group: Union[str, None] = None


@dataclass
class NetworkConfig:
    policy_hidden_layer_sizes: List[int] = field(
        default_factory=lambda: [128, 128, 128, 128]
    )
    value_hidden_layer_sizes: List[int] = field(
        default_factory=lambda: [256, 256, 256, 256, 256]
    )


@dataclass
class VisionNetworkConfig(NetworkConfig):
    policy_hidden_layer_sizes: List[int] = field(
        default_factory=lambda: [32, 32, 32, 32]
    )
    value_hidden_layer_sizes: List[int] = field(
        default_factory=lambda: [256, 256, 256, 256, 256]
    )
    encoder_out_size: int = 4
    cheat_vision_aux_output: bool = False
    has_vision_aux_output: bool = False
    vision_aux_output_mlp: bool = True
    vision_aux_output_decoder: bool = False
    vision_aux_output_mlp_output_size: int = 4
    vision_encoder_normalize_output: bool = True
    stop_vision_gradient: bool = False


@dataclass
class RLConfig:
    num_timesteps: int = 15_000_000
    log_training_metrics: bool = True  #unused
    training_metrics_steps: int = 100000  #unused
    num_evals: int = 0  #unused
    save_interval: int = 30  #replaces num_checkpoints
    reward_scaling: float = 0.1  #unused
    episode_length: int = (
        "${int_divide:${env.task_config.max_duration},${env.ctrl_dt}}"  # TODO: check and fix this dependency!
    )  #deprecated (env.task_config.max_duration is used directly)
    clipping_epsilon: float = 0.3
    normalize_observations: bool = True  #unused (could be used though)
    action_repeat: int = 1  #unused (could be linked with "decimation" though)
    unroll_length: int = 10
    num_minibatches: int = 8
    num_epochs_per_update: int = 8
    num_resets_per_eval: int = 1  #unused
    discounting: float = 0.97
    learning_rate: float = 3e-4
    entropy_cost: float = 0.001
    num_envs: int = 4096
    batch_size: int = 128  #deprecated
    num_eval_envs: int = 5  #unused
    max_grad_norm: float = 1.0
    network_factory: NetworkConfig = field(default_factory=lambda: NetworkConfig())  #unused
    load_checkpoint_path: Union[str, None] = None


class VisionModes(str, Enum):
    rgbd = "rgbd"
    rgb = "rgb"
    depth = "depth"
    depth_w_aux_task = "depth_w_aux_task"


@dataclass
class VisionEnabledConfig:
    enabled: bool = True
    vision_mode: VisionModes = field(default_factory=lambda: VisionModes.rgb)
    gpu_id: int = 0
    render_width: int = 96  #640  #960
    render_height: int = 96  #480  #720
    enabled_geom_groups: List[int] = field(default_factory=lambda: [0, 1, 2])
    observation_cameras: List[int] = field(default_factory=lambda: [0])
    evaluation_cameras: List[int] = field(default_factory=lambda: [0, 1])
    use_rasterizer: bool = False
    num_train_envs: int = 1024  #do not override, use RLConfig.num_envs instead!
    num_eval_envs: int = 8  #do not override, use RLConfig.num_eval_envs instead!


@dataclass
class VisionDisabledConfig:
    enabled: bool = False


defaults = [
    {"wandb": "enabled"},
    {"vision": "disabled"},
    {"env": "pointing"},
    {"rl": "rl_config"},
    {"run": "run"},
    {"rl/network_factory": "${select_network:${vision}}"},
]

@dataclass
class RunConfig:
    seed: int = 0
    play_only: bool = False
    use_tb: bool = False
    rscope_envs: Union[int, None] = None
    deterministic_rscope: bool = True
    domain_randomization: bool = False
    suffix: Union[str, None] = None
    local_plotting: bool = False
    log_wandb_videos: bool = True
    eval_episodes: int = 10
    eval_seed: int = 123
    using_gradio: bool = False


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    wandb: Any = MISSING
    vision: Any = MISSING
    env: BaseEnvConfig = MISSING
    rl: RLConfig = MISSING
    run: RunConfig = MISSING

@dataclass
class UniversalConfig(Config):
    defaults: List[Any] = field(
        default_factory=lambda: [
            {"wandb": "enabled"},
            {"vision": "disabled"},
            {"env": "universal"},
            {"rl": "rl_config"},
            {"run": "run"},
            {"rl/network_factory": "${select_network:${vision}}"},
            {"env/task_config/targets": "default"}, 
        ]
    )

########################
### POINTING CONFIGS ###
########################

@dataclass
class ReachSettings:
    ref_site: str = "humphant"
    ee_site: str = "fingertip"
    target_origin_rel: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

@dataclass
class PointingTaskConfig:
    """
    Default config for pointing task.
    """
    reach_settings: ReachSettings = field(default_factory=lambda: ReachSettings())
    obs_keys: List[str] = field(
        default_factory=lambda: [
            "qpos",
            "qvel",
            "qacc",
            "ee_pos",
            "act",
        ]
    )
    omni_keys: List[str] = field(
        default_factory=lambda: ["target_pos", "target_radius"]
    )
    weighted_reward_keys: Dict[str, float] = field(
        default_factory=lambda: {
            "distance": 1,
            "bonus": 8,
        }
    )
    distance_metric: float = 10.0
    max_duration: float = 4.0
    dwell_duration: float = 0.25
    max_trials: int = 1
    reset_type: str = "range_uniform"
    using_vision_domain_randomisation: bool = "${run.domain_randomization}"

@dataclass
class PointingEnvConfig(BaseEnvConfig):
    env_name: str = "MyoUserPointing"
    model_path: str = ("myosuite/envs/myo/assets/arm/mobl_arms_index_reaching_myouser.xml")
    task_config: PointingTaskConfig = field(default_factory=lambda: PointingTaskConfig())


#########################
### TRACKING CONFIGS ###
#########################

@dataclass
class TrackingTaskConfig(PointingTaskConfig):
    planar_x: bool = True
    num_components: int = 5
    min_amplitude: float = 1
    max_amplitude: float = 5
    min_frequency: float = 0.0
    max_frequency: float = 0.5
    max_episode_steps: int = 40
    omni_keys: List[str] = field(
        default_factory=lambda: [
            "target_pos",
            "prev_target_pos",
            "prev_prev_target_pos",
            "target_radius",
        ]
    )


@dataclass
class TrackingEnvConfig(BaseEnvConfig):
    env_name: str = "MyoUserTracking"
    model_path: str = (
        "myosuite/envs/myo/assets/arm/mobl_arms_index_reaching_myouser.xml"
    )
    task_config: TrackingTaskConfig = field(
        default_factory=lambda: TrackingTaskConfig()
    )


######################
### TARGET CONFIGS ###
######################

@dataclass
class IndividualTargetConfig:
    name: str = MISSING
    rgb: List[float] = MISSING

@dataclass
class TargetsConfig:

    @property
    def targets(self):
        return [getattr(self, f"target_{i}") for i in range(self.num_targets)]

@dataclass
class PointingTarget(IndividualTargetConfig):
    # penetrable: bool = False
    name: str = "pointing_target"
    # Position can either be a 3d vector or a 2 x list of 3d vectors specifying the min and max values for each dimension
    position: List[List[float]] = field(
        default_factory=lambda: [[0.225, -0.1, -0.3], [0.35, 0.1, 0.3]]
    )
    shape: str = "sphere"
    # Size can either be a single value or a list of 2 values specifying the min and max values
    size: List[float] = field(default_factory=lambda: [0.05, 0.15])
    site_pos: List[float] = field(default_factory=lambda: [0, 0, 0.01])  #deprecated, should be removed
    # Any rewards received when inside the target
    reward_incentive: float = 0.0
    completion_bonus: float = 0.0
    dwell_duration: float = 0.25
    rgb: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])


@dataclass
class ButtonTarget(IndividualTargetConfig):
    position: List[List[float]] = MISSING
    name: str = "button_target"
    size: List[List[float]] = field(default_factory=lambda: [[0.025, 0.025, 0.01], [0.025, 0.025, 0.01]])
    site_pos: List[float] = field(default_factory=lambda: [0, 0, 0.01])
    geom_margin: float = 0.0
    completion_bonus: float = 0.0
    min_touch_force: float = 1.0
    rgb: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    euler: List[float] = field(default_factory=lambda: [0, -0.79, 0])

@dataclass
class SingleTargetConfig(TargetsConfig):
    target_0: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]]
        )
    )
    num_targets: int = 1


@dataclass
class DefaultTargetsConfig(TargetsConfig):
    target_0: PointingTarget = field(default_factory=lambda: PointingTarget())
    target_1: ButtonTarget = field(
        default_factory=lambda: ButtonTarget(
            position=[[0.41, -0.05, -0.16], [0.41, -0.05, -0.16]], rgb=[0.8, 0.1, 0.8]
        )
    )
    target_2: PointingTarget = field(
        default_factory=lambda: PointingTarget(rgb=[0.4, 0.0, 1.0])
    )
    target_3: ButtonTarget = field(
        default_factory=lambda: ButtonTarget(
            position=[[0.41, 0.07, -0.16], [0.41, 0.07, -0.16]], rgb=[0.1, 0.8, 0.8]
        )
    )
    target_4: PointingTarget = field(
        default_factory=lambda: PointingTarget(rgb=[0.0, 1.0, 0.4])
    )
    target_5: ButtonTarget = field(
        default_factory=lambda: ButtonTarget(
            position=[[0.50, -0.05, -0.06], [0.50, -0.05, -0.06]], rgb=[0.4, 1.0, 0.2]
        )
    )
    target_6: PointingTarget = field(
        default_factory=lambda: PointingTarget(rgb=[1.0, 1.0, 0.0])
    )
    target_7: ButtonTarget = field(
        default_factory=lambda: ButtonTarget(
            position=[[0.50, 0.07, -0.06], [0.50, 0.07, -0.06]], rgb=[0.1, 0.1, 0.1]
        )
    )
    num_targets: int = 8


@dataclass
class TenPointingTargetsConfig(TargetsConfig):
    num_targets: int = 10
    target_0: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.8, 0.1, 0.1]
        )
    )
    target_1: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.1, 0.8, 0.1]
        )
    )
    target_2: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.1, 0.1, 0.8]
        )
    )
    target_3: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.8, 0.8, 0.1]
        )
    )
    target_4: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.4, 0.4, 0.4]
        )
    )
    target_5: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.4, 0.1, 0.4]
        )
    )
    target_6: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.1, 0.4, 0.4]
        )
    )
    target_7: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.4, 0.4, 0.1]
        )
    )
    target_8: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.1, 0.1, 0.1]
        )
    )
    target_9: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.8, 0.8, 0.8]
        )
    )


########################
### UNIVERSAL CONFIG ###
########################

@dataclass
class UniversalTaskConfig:
    reach_settings: ReachSettings = field(default_factory=lambda: ReachSettings())
    obs_keys: List[str] = field(
        default_factory=lambda: [
            "qpos",
            "qvel",
            "qacc",
            "ee_pos",
            "act",
        ]
    )
    omni_keys: List[str] = field(
        default_factory=lambda: [
            "target_pos",
            "target_size",
            "phase_progress",
            "dwell_fraction",
        ]
    )
    weighted_reward_keys: Dict[str, float] = field(
        default_factory=lambda: {
            "distance": 1,
            "phase_bonus": 0,
            "done": 10,
            "neural_effort": 0.000,
            "jac_effort": 0,
        }
    )
    distance_metric: float = 10.0
    max_duration: float = 6.0
    dwell_duration: float = 0.25
    max_trials: int = 1
    reset_type: str = "range_uniform"
    targets: TargetsConfig = MISSING
    show_all_targets: bool = True
    target_init_seed: int = 1
    exponential_distance_reward: bool = False
    enable_extra_dist: bool = True


@dataclass
class UniversalEnvConfig(BaseEnvConfig):
    env_name: str = "MyoUserUniversal"
    model_path: str = (
        "myosuite/envs/myo/assets/arm/mobl_arms_index_universal_myouser.xml"
    )
    task_config: UniversalTaskConfig = field(
        default_factory=lambda: UniversalTaskConfig()
    )


def create_target_config_class(num_targets: int) -> Type:
    """Create a dataclass with the specified number of target fields."""
    fields = [
        (f"target_{i}", IndividualTargetConfig, MISSING) for i in range(num_targets)
    ]
    fields.append(("num_targets", int, num_targets))

    num2str = [
        "",
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
        "Six",
        "Seven",
        "Eight",
        "Nine",
        "Ten",
    ]

    class_name = f"{num2str[num_targets]}TargetConfig"

    return make_dataclass(
        class_name,
        fields,
        frozen=False,
        bases=(TargetsConfig,),
    )


# # Usage
OneTargetConfig = create_target_config_class(1)
TwoTargetConfig = create_target_config_class(2)
ThreeTargetConfig = create_target_config_class(3)
FourTargetConfig = create_target_config_class(4)
FiveTargetConfig = create_target_config_class(5)
SixTargetConfig = create_target_config_class(6)
SevenTargetConfig = create_target_config_class(7)
EightTargetConfig = create_target_config_class(8)
NineTargetConfig = create_target_config_class(9)
TenTargetConfig = create_target_config_class(10)


LIST_CONFIGS = [
    (1, "one", OneTargetConfig),
    (2, "two", TwoTargetConfig),
    (3, "three", ThreeTargetConfig),
    (4, "four", FourTargetConfig),
    (5, "five", FiveTargetConfig),
    (6, "six", SixTargetConfig),
    (7, "seven", SevenTargetConfig),
    (8, "eight", EightTargetConfig),
    (9, "nine", NineTargetConfig),
    (10, "ten", TenTargetConfig),
    (7, "default", DefaultTargetsConfig),
    (10, "ten_pointing", TenPointingTargetsConfig),
    (1, "single", SingleTargetConfig),
]