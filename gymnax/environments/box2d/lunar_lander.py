"""JAX2D implementation of Gymnasium's LunarLander."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces
from jax2d.engine import PhysicsEngine, create_empty_sim
from jax2d.scene import (
    add_polygon_to_scene,
    add_rectangle_to_scene,
    add_revolute_joint_to_scene,
)
from jax2d.sim_state import SimParams, StaticSimParams, SimState

FPS = 50
SCALE = 30.0

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0

LANDER_POLY = jnp.array(
    [
        (-14.0, +17.0),
        (-17.0, 0.0),
        (-17.0, -10.0),
        (+17.0, -10.0),
        (+17.0, 0.0),
        (+14.0, +17.0),
    ],
    dtype=jnp.float32,
)

LEG_AWAY = 20.0
LEG_DOWN = 18.0
LEG_W = 2.0
LEG_H = 8.0
LEG_SPRING_TORQUE = 40.0

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0
MAIN_ENGINE_Y_LOCATION = 4.0

VIEWPORT_W = 600
VIEWPORT_H = 400

CHUNKS = 11
GROUND_THICKNESS = 0.2


@struct.dataclass
class EnvState(environment.EnvState):
    sim_state: SimState
    prev_shaping: jax.Array
    has_prev_shaping: jax.Array
    leg_contacts: jax.Array
    game_over: jax.Array
    wind_idx: jax.Array
    torque_idx: jax.Array
    helipad_y: jax.Array
    initial_force: jax.Array
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    gravity: float = -10.0
    enable_wind: bool = False
    wind_power: float = 15.0
    turbulence_power: float = 1.5
    initial_random_force_scale: float = INITIAL_RANDOM
    use_fixed_terrain: bool = False
    fixed_height: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros((CHUNKS + 1,), dtype=jnp.float32)
    )
    max_steps_in_episode: int = 1000
    sleep_velocity_threshold: float = 0.01
    sleep_angular_velocity_threshold: float = 0.01


def _segment_to_box(p1: jax.Array, p2: jax.Array, thickness: float) -> tuple[jax.Array, jax.Array, float]:
    delta = p2 - p1
    length = jnp.linalg.norm(delta)
    center = (p1 + p2) * 0.5
    angle = jnp.arctan2(delta[1], delta[0])
    half = jnp.array([length * 0.5, thickness * 0.5], dtype=jnp.float32)
    vertices = jnp.array(
        [
            [-half[0], half[1]],
            [half[0], half[1]],
            [half[0], -half[1]],
            [-half[0], -half[1]],
        ],
        dtype=jnp.float32,
    )
    return center, vertices, angle


def _apply_impulse(
    sim_state: SimState, polygon_index: int, impulse: jax.Array, point: jax.Array
) -> SimState:
    poly = sim_state.polygon
    inv_mass = poly.inverse_mass[polygon_index]
    inv_inertia = poly.inverse_inertia[polygon_index]
    r = point - poly.position[polygon_index]
    dv = impulse * inv_mass
    drv = jnp.cross(r, impulse) * inv_inertia
    new_vel = poly.velocity.at[polygon_index].add(dv)
    new_ang = poly.angular_velocity.at[polygon_index].add(drv)
    return sim_state.replace(
        polygon=poly.replace(velocity=new_vel, angular_velocity=new_ang)
    )


def _apply_force(
    sim_state: SimState, polygon_index: int, force: jax.Array, torque: jax.Array, dt: float
) -> SimState:
    poly = sim_state.polygon
    inv_mass = poly.inverse_mass[polygon_index]
    inv_inertia = poly.inverse_inertia[polygon_index]
    dv = force * inv_mass * dt
    drv = torque * inv_inertia * dt
    new_vel = poly.velocity.at[polygon_index].add(dv)
    new_ang = poly.angular_velocity.at[polygon_index].add(drv)
    return sim_state.replace(
        polygon=poly.replace(velocity=new_vel, angular_velocity=new_ang)
    )


class LunarLander(environment.Environment[EnvState, EnvParams]):
    """JAX2D implementation of Gymnasium's LunarLander."""

    def __init__(self, continuous: bool = False):
        super().__init__()
        self.continuous = continuous
        self.num_ground_segments = CHUNKS - 1
        self.lander_index = self.num_ground_segments
        self.leg_indices = jnp.array(
            [self.num_ground_segments + 1, self.num_ground_segments + 2], dtype=jnp.int32
        )
        self.leg_indices_list = [
            self.num_ground_segments + 1,
            self.num_ground_segments + 2,
        ]

        self.static_sim_params = StaticSimParams(
            num_polygons=16,
            num_circles=2,
            num_joints=2,
            num_thrusters=1,
            max_polygon_vertices=6,
            num_static_fixated_polys=self.num_ground_segments,
        )
        self.engine = PhysicsEngine(self.static_sim_params)
        self.sim_params = SimParams(dt=1.0 / FPS)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def _build_sim_state(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[SimState, jax.Array, jax.Array]:
        key_height, key_force, key_wind = jax.random.split(key, 3)

        w = VIEWPORT_W / SCALE
        h = VIEWPORT_H / SCALE

        height = jax.lax.select(
            jnp.array(params.use_fixed_terrain),
            params.fixed_height,
            jax.random.uniform(key_height, shape=(CHUNKS + 1,), minval=0.0, maxval=h / 2.0),
        )
        helipad_y = h / 4.0
        height = height.at[CHUNKS // 2 - 2 : CHUNKS // 2 + 3].set(helipad_y)

        idx = jnp.arange(CHUNKS)
        prev_idx = (idx - 1) % (CHUNKS + 1)
        next_idx = idx + 1
        smooth_y = 0.33 * (height[prev_idx] + height[idx] + height[next_idx])

        chunk_x = jnp.arange(CHUNKS, dtype=jnp.float32) * (w / (CHUNKS - 1))

        sim_state = create_empty_sim(
            self.static_sim_params,
            add_floor=False,
            add_walls_and_ceiling=False,
        )

        for i in range(self.num_ground_segments):
            p1 = jnp.array([chunk_x[i], smooth_y[i]], dtype=jnp.float32)
            p2 = jnp.array([chunk_x[i + 1], smooth_y[i + 1]], dtype=jnp.float32)
            center, vertices, angle = _segment_to_box(p1, p2, GROUND_THICKNESS)
            sim_state, _ = add_polygon_to_scene(
                sim_state,
                self.static_sim_params,
                center,
                vertices,
                4,
                rotation=angle,
                density=1.0,
                friction=0.1,
                restitution=0.0,
                fixated=True,
            )

        lander_pos = jnp.array([w / 2.0, h], dtype=jnp.float32)
        lander_vertices = LANDER_POLY / SCALE
        sim_state, _ = add_polygon_to_scene(
            sim_state,
            self.static_sim_params,
            lander_pos,
            lander_vertices,
            6,
            rotation=0.0,
            velocity=jnp.zeros(2, dtype=jnp.float32),
            angular_velocity=0.0,
            density=5.0,
            friction=0.1,
            restitution=0.0,
            fixated=False,
        )

        leg_offsets = jnp.array([-1.0, 1.0], dtype=jnp.float32)
        for i, leg_sign in enumerate(leg_offsets):
            leg_pos = jnp.array(
                [w / 2.0 - leg_sign * LEG_AWAY / SCALE, h], dtype=jnp.float32
            )
            sim_state, _ = add_rectangle_to_scene(
                sim_state,
                self.static_sim_params,
                leg_pos,
                jnp.array([LEG_W / SCALE, LEG_H / SCALE], dtype=jnp.float32),
                rotation=leg_sign * 0.05,
                density=1.0,
                friction=0.1,
                restitution=0.0,
                fixated=False,
            )

        for i, leg_sign in enumerate(leg_offsets):
            min_angle = jax.lax.select(leg_sign < 0, 0.9 - 0.5, -0.9)
            max_angle = jax.lax.select(leg_sign < 0, 0.9, -0.9 + 0.5)
            motor_speed = 0.3 * leg_sign / self.sim_params.base_motor_speed
            motor_power = LEG_SPRING_TORQUE / self.sim_params.base_motor_power
            sim_state, _ = add_revolute_joint_to_scene(
                sim_state,
                self.static_sim_params,
                self.lander_index,
                self.leg_indices_list[i],
                jnp.array([0.0, 0.0], dtype=jnp.float32),
                jnp.array([leg_sign * LEG_AWAY / SCALE, LEG_DOWN / SCALE], dtype=jnp.float32),
                motor_on=True,
                motor_speed=motor_speed,
                motor_power=motor_power,
                has_joint_limits=True,
                min_rotation=min_angle,
                max_rotation=max_angle,
            )

        force = jax.random.uniform(
            key_force, shape=(2,), minval=-1.0, maxval=1.0
        ) * params.initial_random_force_scale

        key_wind, key_torque = jax.random.split(key_wind)
        wind_idx = jax.random.randint(key_wind, shape=(), minval=-9999, maxval=9999)
        torque_idx = jax.random.randint(key_torque, shape=(), minval=-9999, maxval=9999)
        use_wind = jnp.array(params.enable_wind)
        wind_idx = jax.lax.select(use_wind, wind_idx, jnp.array(0, dtype=jnp.int32))
        torque_idx = jax.lax.select(use_wind, torque_idx, jnp.array(0, dtype=jnp.int32))

        sim_state = sim_state.replace(gravity=jnp.array([0.0, params.gravity], dtype=jnp.float32))
        return sim_state, helipad_y, (force, wind_idx, torque_idx)

    def _compute_contacts(self, rr_manifolds) -> tuple[jax.Array, jax.Array]:
        pairs = self.engine.poly_poly_pairs
        rr_active = rr_manifolds.active

        is_ground_a = pairs[:, 0] < self.num_ground_segments
        is_ground_b = pairs[:, 1] < self.num_ground_segments

        lander_hit = ((pairs[:, 0] == self.lander_index) & is_ground_b) | (
            (pairs[:, 1] == self.lander_index) & is_ground_a
        )
        lander_contact = jnp.any(rr_active & lander_hit)

        def _leg_contact(leg_index):
            leg_hit = ((pairs[:, 0] == leg_index) & is_ground_b) | (
                (pairs[:, 1] == leg_index) & is_ground_a
            )
            return jnp.any(rr_active & leg_hit)

        leg_contacts = jax.vmap(_leg_contact)(self.leg_indices).astype(jnp.float32)
        return lander_contact, leg_contacts

    def _state_from_sim(
        self, sim_state: SimState, helipad_y: jax.Array, leg_contacts: jax.Array
    ) -> jax.Array:
        pos = sim_state.polygon.position[self.lander_index]
        vel = sim_state.polygon.velocity[self.lander_index]
        angle = sim_state.polygon.rotation[self.lander_index]
        ang_vel = sim_state.polygon.angular_velocity[self.lander_index]

        w = VIEWPORT_W / SCALE
        h = VIEWPORT_H / SCALE

        state = jnp.array(
            [
                (pos[0] - w / 2.0) / (w / 2.0),
                (pos[1] - (helipad_y + LEG_DOWN / SCALE)) / (h / 2.0),
                vel[0] * (w / 2.0) / FPS,
                vel[1] * (h / 2.0) / FPS,
                angle,
                20.0 * ang_vel / FPS,
                leg_contacts[0],
                leg_contacts[1],
            ],
            dtype=jnp.float32,
        )
        return state

    def _apply_engines(
        self, key: jax.Array, sim_state: SimState, action: jax.Array
    ) -> tuple[SimState, jax.Array, jax.Array]:
        lander_pos = sim_state.polygon.position[self.lander_index]
        lander_angle = sim_state.polygon.rotation[self.lander_index]
        tip = jnp.array([jnp.sin(lander_angle), jnp.cos(lander_angle)], dtype=jnp.float32)
        side = jnp.array([-tip[1], tip[0]], dtype=jnp.float32)

        dispersion = jax.random.uniform(
            key, shape=(2,), minval=-1.0 / SCALE, maxval=1.0 / SCALE
        )

        if self.continuous:
            action = jnp.clip(action, -1.0, 1.0)
            main_fire = action[0] > 0.0
            side_fire = jnp.abs(action[1]) > 0.5
            m_power = jax.lax.select(
                main_fire, (jnp.clip(action[0], 0.0, 1.0) + 1.0) * 0.5, 0.0
            )
            direction = jnp.sign(action[1])
            s_power = jax.lax.select(
                side_fire, jnp.clip(jnp.abs(action[1]), 0.5, 1.0), 0.0
            )
        else:
            action = action.astype(jnp.int32)
            main_fire = action == 2
            side_fire = (action == 1) | (action == 3)
            m_power = jax.lax.select(main_fire, 1.0, 0.0)
            direction = action - 2
            s_power = jax.lax.select(side_fire, 1.0, 0.0)

        ox = tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2.0 * dispersion[0]) + side[0] * dispersion[1]
        oy = -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2.0 * dispersion[0]) - side[1] * dispersion[1]
        impulse_pos = lander_pos + jnp.array([ox, oy], dtype=jnp.float32)
        impulse = jnp.array(
            [-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power],
            dtype=jnp.float32,
        )
        sim_state = _apply_impulse(sim_state, int(self.lander_index), impulse, impulse_pos)

        ox = tip[0] * dispersion[0] + side[0] * (3.0 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
        oy = -tip[1] * dispersion[0] - side[1] * (3.0 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
        impulse_pos = lander_pos + jnp.array(
            [ox - tip[0] * 17.0 / SCALE, oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE],
            dtype=jnp.float32,
        )
        impulse = jnp.array(
            [-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power],
            dtype=jnp.float32,
        )
        sim_state = _apply_impulse(sim_state, int(self.lander_index), impulse, impulse_pos)

        return sim_state, m_power, s_power

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        sim_state = state.sim_state

        key_dispersion, key_wind = jax.random.split(key)

        sim_state = _apply_force(
            sim_state,
            int(self.lander_index),
            state.initial_force,
            jnp.array(0.0, dtype=jnp.float32),
            self.sim_params.dt,
        )
        initial_force = jnp.zeros(2, dtype=jnp.float32)

        apply_wind = jnp.logical_and(
            jnp.array(params.enable_wind),
            jnp.logical_not(jnp.any(state.leg_contacts > 0)),
        )

        def _do_wind(args):
            sim_state, wind_idx, torque_idx = args
            wind_mag = jnp.tanh(
                jnp.sin(0.02 * wind_idx) + jnp.sin(jnp.pi * 0.01 * wind_idx)
            ) * params.wind_power
            torque_mag = jnp.tanh(
                jnp.sin(0.02 * torque_idx) + jnp.sin(jnp.pi * 0.01 * torque_idx)
            ) * params.turbulence_power
            sim_state = _apply_force(
                sim_state,
                self.lander_index,
                jnp.array([wind_mag, 0.0], dtype=jnp.float32),
                torque_mag,
                self.sim_params.dt,
            )
            return sim_state, wind_idx + 1, torque_idx + 1

        def _no_wind(args):
            return args

        sim_state, wind_idx, torque_idx = jax.lax.cond(
            apply_wind,
            _do_wind,
            _no_wind,
            (sim_state, state.wind_idx, state.torque_idx),
        )

        sim_state, m_power, s_power = self._apply_engines(
            key_dispersion, sim_state, jnp.asarray(action)
        )

        motor_actions = sim_state.joint.motor_on.astype(jnp.float32)
        actions = jnp.concatenate(
            [motor_actions, jnp.zeros((self.static_sim_params.num_thrusters,), dtype=jnp.float32)]
        )

        sim_state, (rr_manifolds, _, _) = self.engine.step(
            sim_state, self.sim_params, actions
        )

        lander_contact, leg_contacts = self._compute_contacts(rr_manifolds)
        game_over = lander_contact

        obs_state = self._state_from_sim(sim_state, state.helipad_y, leg_contacts)

        shaping = (
            -100.0 * jnp.sqrt(obs_state[0] * obs_state[0] + obs_state[1] * obs_state[1])
            - 100.0 * jnp.sqrt(obs_state[2] * obs_state[2] + obs_state[3] * obs_state[3])
            - 100.0 * jnp.abs(obs_state[4])
            + 10.0 * obs_state[6]
            + 10.0 * obs_state[7]
        )
        reward = jax.lax.select(state.has_prev_shaping, shaping - state.prev_shaping, 0.0)
        reward = reward - m_power * 0.30 - s_power * 0.03

        landed = (
            (leg_contacts[0] > 0)
            & (leg_contacts[1] > 0)
            & (jnp.linalg.norm(sim_state.polygon.velocity[self.lander_index]) < params.sleep_velocity_threshold)
            & (jnp.abs(sim_state.polygon.angular_velocity[self.lander_index]) < params.sleep_angular_velocity_threshold)
        )

        terminated = jnp.array(False)
        reward = jax.lax.select(game_over | (jnp.abs(obs_state[0]) >= 1.0), -100.0, reward)
        terminated = terminated | game_over | (jnp.abs(obs_state[0]) >= 1.0)
        reward = jax.lax.select(landed & jnp.logical_not(terminated), 100.0, reward)
        terminated = terminated | landed

        next_state = EnvState(
            sim_state=sim_state,
            prev_shaping=shaping,
            has_prev_shaping=jnp.array(True),
            leg_contacts=leg_contacts,
            game_over=game_over,
            wind_idx=wind_idx,
            torque_idx=torque_idx,
            helipad_y=state.helipad_y,
            initial_force=initial_force,
            time=state.time + 1,
        )

        done = self.is_terminal(next_state, params) | terminated
        return (
            jax.lax.stop_gradient(obs_state),
            jax.lax.stop_gradient(next_state),
            reward,
            done,
            {"discount": self.discount(next_state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        sim_state, helipad_y, (force, wind_idx, torque_idx) = self._build_sim_state(key, params)
        leg_contacts = jnp.zeros((2,), dtype=jnp.float32)
        state = EnvState(
            sim_state=sim_state,
            prev_shaping=jnp.array(0.0, dtype=jnp.float32),
            has_prev_shaping=jnp.array(False),
            leg_contacts=leg_contacts,
            game_over=jnp.array(False),
            wind_idx=wind_idx,
            torque_idx=torque_idx,
            helipad_y=helipad_y,
            initial_force=force,
            time=0,
        )

        key_step = jax.random.fold_in(key, 1)
        reset_action = jnp.array([0.0, 0.0]) if self.continuous else jnp.array(0)
        obs, state, _, _, _ = self.step_env(key_step, state, reset_action, params)
        state = state.replace(time=0)
        return obs, state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        return self._state_from_sim(state.sim_state, state.helipad_y, state.leg_contacts)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        time_limit = state.time >= params.max_steps_in_episode
        return jnp.array(time_limit)

    @property
    def name(self) -> str:
        return "LunarLander-v3"

    @property
    def num_actions(self) -> int:
        return 2 if self.continuous else 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Box | spaces.Discrete:
        if self.continuous:
            return spaces.Box(-1.0, 1.0, (2,), jnp.float32)
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        low = jnp.array(
            [-2.5, -2.5, -10.0, -10.0, -2 * jnp.pi, -10.0, -0.0, -0.0],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [2.5, 2.5, 10.0, 10.0, 2 * jnp.pi, 10.0, 1.0, 1.0],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (8,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "time": spaces.Discrete(params.max_steps_in_episode + 1),
                "prev_shaping": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "leg_contacts": spaces.Box(0.0, 1.0, (2,), jnp.float32),
                "game_over": spaces.Discrete(2),
                "wind_idx": spaces.Box(
                    -jnp.iinfo(jnp.int32).max,
                    jnp.iinfo(jnp.int32).max,
                    (),
                    jnp.int32,
                ),
                "torque_idx": spaces.Box(
                    -jnp.iinfo(jnp.int32).max,
                    jnp.iinfo(jnp.int32).max,
                    (),
                    jnp.int32,
                ),
                "helipad_y": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
            }
        )
