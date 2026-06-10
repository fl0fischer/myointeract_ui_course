import mujoco
import numpy as np
import torch

from omegaconf import MISSING
from typing import Any, Dict, List, Tuple, Type, Union
from dataclasses import dataclass, field, make_dataclass

@dataclass
class IndividualTargetConfig:
    name: str = MISSING
    rgb: List[float] = MISSING


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


def generate_target(target_pos_range: np.ndarray, target_radius_range: np.ndarray, target_coordinates_origin: np.ndarray, rng: np.random.Generator, prev_pos: np.ndarray = None, min_distance: float = 0.1):
    target_pos = generate_target_pos(target_pos_range, target_coordinates_origin, rng, prev_pos=prev_pos, min_distance=min_distance)
    target_size = generate_target_size(target_radius_range, rng)
    return target_pos, target_size

def generate_target_pos(target_pos_range, target_coordinates_origin, rng: np.random.Generator, prev_pos: np.ndarray = None, min_distance: float = 0.1):
    if prev_pos is None:
        prev_pos = np.array([-999.0, -999.0, -999.0])
    new_pos = prev_pos.copy()
    while _unsatisfied_dist_constr(new_pos, prev_pos, min_distance=min_distance):
        prev_pos = new_pos
        new_pos = _target_pos_sampler(target_pos_range, target_coordinates_origin, rng)
    return new_pos

def _unsatisfied_dist_constr(new_pos, prev_pos, min_distance=0.1):
    # distance = np.linalg.norm(
    #     new_pos - prev_pos, axis=-1
    # )
    # return distance < min_distance
    distance = np.abs(new_pos - prev_pos)
    return np.any(distance < min_distance)

def _target_pos_sampler(target_pos_range, target_coordinates_origin, rng: np.random.Generator):
    sampled_pos = rng.random(3) * (target_pos_range[1] - target_pos_range[0]) + target_pos_range[0]
    sampled_pos = target_coordinates_origin + sampled_pos
    return sampled_pos

def generate_target_size(target_radius_range, rng: np.random.Generator):
    target_size = rng.random(3) * (target_radius_range[1] - target_radius_range[0]) + target_radius_range[0]
    return target_size


def add_sphere_to_spec(
    spec: mujoco.MjSpec, target_cfg: PointingTarget, target_id: int, target_coordinates_origin: np.ndarray = np.zeros(3), seed: int = None
):
    ## TODO: unify usage of torch vs numpy
    rng = np.random.default_rng(seed)
    target_body_name = f"body_target_{target_id}"
    target_geom_name = f"geom_target_{target_id}"
    target_site_name = f"site_target_{target_id}"
    target_sensor_name = f"sensor_target_{target_id}"

    worldbody = spec.body("root-variant")
    if worldbody is None:
        worldbody = spec.worldbody
    else:
        # worldbody.gravcomp = 1  #counteract gravity to keep the target in place  #TODO: does not work?
        root_free_joint = worldbody.add_joint(name="root-variant-joint", type=mujoco.mjtJoint.mjJNT_FREE)
        root_free_joint.damping[0] = 1e+10   # set huge damping on the free joint to keep both the body and the target added below in place
    target_pos, target_size = generate_target(np.array(target_cfg.position), np.array(target_cfg.size), target_coordinates_origin, rng)
    target_body = worldbody.add_body(name=target_body_name, pos=target_pos)
    rgba = np.ones(4)
    rgba[:3] = target_cfg.rgb
    target_size = np.ones(3) * target_size  ##TODO: deprecated -- remove
    target_geom = target_body.add_geom(
        name=target_geom_name, pos=np.zeros(3), size=target_size, rgba=rgba
    )
    # print(f"Added target {target_geom_name} to spec")

    #### only required for consistency with add_button_to_spec
    target_site = target_body.add_site(
        name=target_site_name,
        type=mujoco._enums.mjtGeom(6),
        pos=target_cfg.site_pos,
        rgba=rgba,
        size=0.001 * np.ones(3),
    )
    sensor = spec.add_sensor(
        name=target_sensor_name,
        type=mujoco.mjtSensor.mjSENS_TOUCH,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=target_site_name,
    )

    return spec


def add_button_to_spec(
    spec: mujoco.MjSpec, target_cfg: ButtonTarget, target_id: int, target_coordinates_origin: np.ndarray = np.zeros(3), seed: int = None
):
    rng = np.random.default_rng(seed)
    target_body_name = f"body_target_{target_id}"
    target_geom_name = f"geom_target_{target_id}"
    target_site_name = f"site_target_{target_id}"
    target_sensor_name = f"sensor_target_{target_id}"

    worldbody = spec.worldbody
    target_pos, target_size = generate_target(np.array(target_cfg.position), np.array(target_cfg.size), target_coordinates_origin, rng)
    target_body = worldbody.add_body(
        name=target_body_name, pos=target_pos, euler=target_cfg.euler
    )
    rgba = np.ones(4)
    rgba[:3] = target_cfg.rgb
    target_geom = target_body.add_geom(
        name=target_geom_name,
        type=mujoco._enums.mjtGeom(6),
        size=target_size,
        margin=target_cfg.geom_margin,
        rgba=rgba,
        contype=1,
        conaffinity=1,
    )
    target_site = target_body.add_site(
        name=target_site_name,
        type=mujoco._enums.mjtGeom(6),
        pos=target_cfg.site_pos,
        rgba=rgba,
        size=target_size,
    )
    # print(f"Added target {target_geom_name} to spec")
    # Add touch sensor for the button
    sensor = spec.add_sensor(
        name=target_sensor_name,
        type=mujoco.mjtSensor.mjSENS_TOUCH,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=target_site_name,
    )
    # print(f"Added sensor {target_sensor_name} to spec")
    return spec