"""
This script contains an older task used to train a single agent to play soccer.
Modified from dm_control/locomotion/soccer/task.py, which is licensed under the following terms:
"""


# Copyright 2019 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

""""A task where players play a soccer game."""


from dm_control import composer
from dm_control.locomotion.soccer import initializers
from dm_control.locomotion.soccer import observables as observables_lib
from dm_control.locomotion.soccer import soccer_ball
from dm_control.locomotion.soccer import _make_players
from dm_control.locomotion.soccer import pitch
from dm_control.locomotion.soccer import WalkerType
from dm_control.locomotion.soccer import team
from dm_env import specs
import numpy as np
import math

from dm_control.utils import rewards


_THROW_IN_BALL_Z = 0.5


def _disable_geom_contacts(entities):
  for entity in entities:
    mjcf_model = entity.mjcf_model
    for geom in mjcf_model.find_all("geom"):
      geom.set_attributes(contype=0)


class ReachBall(composer.Task):
  """A task where two teams of walkers play soccer."""

  def __init__(
      self,
      players,
      arena,
      ball=None,
      initializer=None,
      observables=None,
      disable_walker_contacts=False,
      nconmax_per_player=200,
      njmax_per_player=400,
      control_timestep=0.025,
      tracking_cameras=(),
  ):
    """Construct an instance of soccer.Task.

    This task implements the high-level game logic of multi-agent MuJoCo soccer.

    Args:
      players: a sequence of `soccer.Player` instances, representing
        participants to the game from both teams.
      arena: an instance of `soccer.Pitch`, implementing the physical geoms and
        the sensors associated with the pitch.
      ball: optional instance of `soccer.SoccerBall`, implementing the physical
        geoms and sensors associated with the soccer ball. If None, defaults to
        using `soccer_ball.SoccerBall()`.
      initializer: optional instance of `soccer.Initializer` that initializes
        the task at the start of each episode. If None, defaults to
        `initializers.UniformInitializer()`.
      observables: optional instance of `soccer.ObservablesAdder` that adds
        observables for each player. If None, defaults to
        `observables.CoreObservablesAdder()`.
      disable_walker_contacts: if `True`, disable physical contacts between
        players.
      nconmax_per_player: allocated maximum number of contacts per player. It
        may be necessary to increase this value if you encounter errors due to
        `mjWARN_CONTACTFULL`.
      njmax_per_player: allocated maximum number of scalar constraints per
        player. It may be necessary to increase this value if you encounter
        errors due to `mjWARN_CNSTRFULL`.
      control_timestep: control timestep of the agent.
      tracking_cameras: a sequence of `camera.MultiplayerTrackingCamera`
        instances to track the players and ball.
    """
    self.arena = arena
    self.players = players

    self._initializer = initializer or initializers.UniformInitializer()
    self._observables = observables or observables_lib.CoreObservablesAdder()

    if disable_walker_contacts:
      _disable_geom_contacts([p.walker for p in self.players])

    # Create ball and attach ball to arena.
    self.ball = ball or soccer_ball.SoccerBall()
    self.arena.add_free_entity(self.ball)
    self.arena.register_ball(self.ball)

    # Register soccer ball contact tracking for players.
    for player in self.players:
      player.walker.create_root_joints(self.arena.attach(player.walker))
      self.ball.register_player(player)
      # Add per-walkers observables.
      self._observables(self, player)

    self._tracking_cameras = tracking_cameras

    self.set_timesteps(
        physics_timestep=0.005, control_timestep=control_timestep)
    self.root_entity.mjcf_model.size.nconmax = nconmax_per_player * len(players)
    self.root_entity.mjcf_model.size.njmax = njmax_per_player * len(players)

  @property
  def observables(self):
    observables = []
    for player in self.players:
      observables.append(
          player.walker.observables.as_dict(fully_qualified=False))
    return observables

  def _throw_in(self, physics, random_state, ball):
    x, y, _ = physics.bind(ball.geom).xpos
    shrink_x, shrink_y = random_state.uniform([0.7, 0.7], [0.9, 0.9])
    ball.set_pose(physics, [x * shrink_x, y * shrink_y, _THROW_IN_BALL_Z])
    ball.set_velocity(
        physics, velocity=np.zeros(3), angular_velocity=np.zeros(3))
    ball.initialize_entity_trackers()

  def _tracked_entity_positions(self, physics):
    """Return a list of the positions of the ball and all players."""
    ball_pos, unused_ball_quat = self.ball.get_pose(physics)
    entity_positions = [ball_pos]
    for player in self.players:
      walker_pos, unused_walker_quat = player.walker.get_pose(physics)
      entity_positions.append(walker_pos)
    return entity_positions

  def after_compile(self, physics, random_state):
    super().after_compile(physics, random_state)
    for camera in self._tracking_cameras:
      camera.after_compile(physics)

  def after_step(self, physics, random_state):
    super().after_step(physics, random_state)
    for camera in self._tracking_cameras:
      camera.after_step(self._tracked_entity_positions(physics))

  def initialize_episode_mjcf(self, random_state):
    self.arena.initialize_episode_mjcf(random_state)

  def initialize_episode(self, physics, random_state):
    self.arena.initialize_episode(physics, random_state)
    for player in self.players:
      player.walker.reinitialize_pose(physics, random_state)

    self._initializer(self, physics, random_state)
    for camera in self._tracking_cameras:
      camera.initialize_episode(self._tracked_entity_positions(physics))

  def _detect_ball_contact(self, physics, player_idx):
    ball_pos = self._tracked_entity_positions(physics)[0]
    player_pos = self._tracked_entity_positions(physics)[player_idx + 1]
    distance = np.linalg.norm(np.array(ball_pos) - np.array(player_pos))
    
    return distance < 0.1


  @property
  def root_entity(self):
    return self.arena

  def get_reward(self, physics):
    """Returns a list of per-player rewards.

    Reward is 

    Note: the observations also contain various environment statistics that may
    be used to derive per-player rewards (as done in
    http://arxiv.org/abs/1902.07151).

    Args:
      physics: An instance of `Physics`.

    Returns:
      A list of 0-dimensional numpy arrays, one per player.
    """
    # scoring_team = self.arena.detected_goal()
    # if not scoring_team:
    #   return [np.zeros((), dtype=np.float32) for _ in self.players]
    field_size = self.arena.size
    rewards_list = []
    entity_positions = self._tracked_entity_positions(physics=physics)
    # for i in range(len(entity_positions[1:])):
    #   dist_xyz_normed = [entity_positions[i+1][0] / field_size[0], entity_positions[i+1][1] / field_size[1]]
    #   reward = np.exp(-(np.linalg.norm(dist_xyz_normed))**2)
    #   rewards.append(np.asarray(reward, dtype=np.float32))
    # print("All pos ", entity_positions)
    # print("Field size ", field_size)
    ####
    ball_pos = entity_positions[0]
    # print("Ball pos ", ball_pos)
    distances_to_ball = [x - ball_pos for x in entity_positions]
    # print("Dist to ball ", distances_to_ball)
    # for dist_xyz in distances_to_ball[1:]:
    #     dist_xyz_normed = [dist_xyz[0] / field_size[0], dist_xyz[1] / field_size[1]]
    #     # print("Dist_xyz normalised ", dist_xyz_normed)
    #     # print("Norm ", np.linalg.norm(dist_xyz))
    #     reward = np.exp(-(np.linalg.norm(dist_xyz_normed))**2)
    #     rewards_list.append(np.asarray(reward, dtype=np.float32))
    ####
    # Best so far
    # for i in range(len(self.players)):
    #   vel_to_ball_obs = self.observables[i]['stats_vel_to_ball']
    #   vel_to_ball = vel_to_ball_obs(physics)
    #   reward = max(0, vel_to_ball)
    #   rewards_list.append(np.asarray(reward, dtype=np.float32))
    ####

    goal_vel = 0.1
    for i in range(len(self.players)):
      if self.players[i].team == team.Team.HOME:
        move_reward = rewards.tolerance(
            self.observables[i]['stats_vel_to_ball'](physics),
            bounds=(float(goal_vel), float('inf')),
            margin=goal_vel,
            value_at_margin=0.1,
            sigmoid='linear')
        
        ball_move_reward = max(0, self.observables[i]['stats_vel_ball_to_goal'](physics))
        
        # move_reward = max(0, self.observables[i]['stats_vel_to_ball'](physics)*0.05)

        # deviation = np.cos(np.deg2rad(0))
        # torso_upright = np.asarray(physics.named.data.xmat['home0/torso', 'zz'])
        # upright_reward = rewards.tolerance(
        #     torso_upright,
        #     bounds=(deviation, float('inf')),
        #     sigmoid='linear',
        #     margin=1 + deviation,
        #     value_at_margin=0)
        
        # contact_reward = 0.0
        # if self._detect_ball_contact(physics, i):
        #   contact_reward = 10.0

        # dist_xy = [distances_to_ball[1][0], distances_to_ball[1][1]]
        # dist_reward = np.exp(-0.05*(np.linalg.norm(dist_xy))**2)

        total_reward = move_reward + ball_move_reward

      else:
        total_reward = 0.0
    
      rewards_list.append(np.asarray(total_reward, dtype=np.float32))


    return rewards_list

  def get_reward_spec(self):
    return [
        specs.Array(name="reward", shape=(), dtype=np.float32)
        for _ in self.players
    ]

  def get_discount(self, physics):
    if self.arena.detected_goal():
      return np.zeros((), np.float32)
    return np.ones((), np.float32)

  def get_discount_spec(self):
    return specs.Array(name="discount", shape=(), dtype=np.float32)

  def should_terminate_episode(self, physics):
    """Returns True if a goal was scored by either team."""
    return self.arena.detected_goal() is not None

  def before_step(self, physics, actions, random_state):
    for player, action in zip(self.players, actions):
      player.walker.apply_action(physics, action, random_state)

    if self.arena.detected_off_court():
      self._throw_in(physics, random_state, self.ball)

  def action_spec(self, physics):
    """Return multi-agent action_spec."""
    return [player.walker.action_spec for player in self.players]


def load_environment(team_size,
         time_limit = 40.0,
         random_state = None,
         disable_walker_contacts = False,
         enable_field_box = False,
         keep_aspect_ratio = True,
         terminate_on_goal = True,
         walker_type = WalkerType.ANT):
    
    goal_size = None
    # min_size = (32, 24)
    min_size = (48, 36)
    max_size = (48, 36)
    ball = soccer_ball.SoccerBall()

    # return composer.Environment(
    #     task=ReachBall(
    #         players = _make_players(team_size, walker_type),
    #         arena = pitch.RandomizedPitch(
    #             min_size = min_size,
    #             max_size = max_size,
    #             keep_aspect_ratio = keep_aspect_ratio,
    #             field_box = enable_field_box,
    #             goal_size = goal_size),
    #         ball = ball,
    #         disable_walker_contacts = disable_walker_contacts),
    #     time_limit = time_limit,
    #     random_state = random_state)

    return composer.Environment(
        task=ReachBall(
            players = _make_players(team_size, walker_type),
            arena = pitch.Pitch(
                size=(48, 36),
                field_box = enable_field_box,
                goal_size = goal_size),
            ball = ball,
            disable_walker_contacts = disable_walker_contacts),
        time_limit = time_limit,
        random_state = random_state)




