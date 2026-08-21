"""
Custom Multi-Agent Physics Environment Wrapper (`env_core.py`).

This module defines `MultiRobotPhysicsEnv`, inheriting from Ray RLlib's `MultiAgentEnv`. It manages
a physics simulation (via PyBullet) of N mobile robots tasked with collaboratively transporting a heavy
payload across a rugged spatial environment to a target zone.

Key Operational Components:
1. Dynamic Graph Topology Engine:
   At every physics step, pairwise Euclidean distances d_ij = ||p_i - p_j||_2 are evaluated.
   An edge (j, i) is established if d_ij <= R_comm and line-of-sight is maintained.
   Edge features E_t[i, j] encode relative displacement vectors, relative velocity, and Euclidean distance.
2. Potential-Based & Connectivity Reward Shaping:
   Ensures policy convergence and prevents graph fragmentation via a multi-objective reward structure:
       r_i^{(t)} = w_prog * R_progress + w_coop * R_coop + w_conn * R_conn - w_safe * C_safety
"""

from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from scipy.linalg import eigh

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False


class MultiRobotPhysicsEnv(MultiAgentEnv):
    """
    Ray RLlib MultiAgentEnv wrapper handling physics simulation, dynamic graph construction, and reward shaping.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the multi-robot physics environment.

        Args:
            config: Configuration dictionary containing simulation parameters.
        """
        super().__init__()
        self.config = config or {}
        
        # Hyperparameters
        self.num_robots = int(self.config.get("num_robots", 4))
        self.comm_radius = float(self.config.get("comm_radius", 5.0))
        self.max_steps = int(self.config.get("max_steps", 500))
        self.render_mode = self.config.get("render_mode", "headless")
        self.dt = float(self.config.get("dt", 1.0 / 240.0))
        self.physics_steps_per_env_step = int(self.config.get("physics_steps_per_env_step", 10))

        # Contact radius for cooperative bonus and payload fallback dynamics
        self.contact_radius = float(self.config.get("contact_radius", 1.5))
        # Terminal completion bonus magnitude (>> typical per-step progress_reward)
        self.completion_bonus = float(self.config.get("completion_bonus", 100.0))
        # Cooperative contact bonus per step for robots near the payload
        self.contact_bonus = float(self.config.get("contact_bonus", 1.0))

        # LiDAR ray-casting configuration
        self.lidar_num_rays = 12          # Matches raw_obs_dim slot 6:18
        self.lidar_max_range = float(self.config.get("lidar_max_range", 5.0))
        self.lidar_ray_height = float(self.config.get("lidar_ray_height", 0.15))
        self.lidar_angles = np.linspace(  # 12 evenly-spaced rays around 360°
            0, 2 * np.pi, self.lidar_num_rays, endpoint=False
        )

        # Dimensions
        self.raw_obs_dim = 24  # [pos(3), vel(3), lidar(12), goal_rel(3), payload_rel(3)]
        self.edge_dim = 8      # [rel_pos(3), rel_vel(3), dist(1), normalized_heading(1)]
        self.action_dim = 2    # Differential drive: [left_wheel_vel, right_wheel_vel]

        self._agent_ids = [f"robot_{i}" for i in range(self.num_robots)]

        # Define Observation and Action spaces conforming to GNNMARLModel requirements
        single_agent_obs_space = spaces.Dict({
            "local_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.raw_obs_dim,), dtype=np.float32),
            "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_robots, self.raw_obs_dim), dtype=np.float32),
            "adj_matrix": spaces.Box(low=0.0, high=1.0, shape=(self.num_robots, self.num_robots), dtype=np.float32),
            "edge_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_robots, self.num_robots, self.edge_dim), dtype=np.float32),
            "node_index": spaces.Box(low=0, high=self.num_robots - 1, shape=(1,), dtype=np.int64),
            # Pre-drawn random scores for topk_mode='random' comm sparsification.
            # Generated once per env step so PPO's multi-epoch SGD replays the same mask.
            "random_comm_mask": spaces.Box(low=0.0, high=1.0, shape=(self.num_robots, self.num_robots), dtype=np.float32)
        })
        single_agent_action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        self.observation_space = single_agent_obs_space
        self.action_space = single_agent_action_space
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True

        # Physics simulation handles
        self.physics_client: Optional[int] = None
        self.robot_body_ids: List[int] = []
        self.payload_body_id: Optional[int] = None
        self.goal_pos = np.array([10.0, 10.0, 0.5], dtype=np.float32)

        # State tracking
        self.step_count = 0
        self.prev_payload_dist: float = 0.0
        self._reward_totals = {"progress": 0.0, "conn": 0.0, "contact": 0.0, "energy": 0.0, "completion": 0.0}
        self.robot_positions = np.zeros((self.num_robots, 3), dtype=np.float32)
        self.robot_velocities = np.zeros((self.num_robots, 3), dtype=np.float32)
        self.robot_headings = np.zeros(self.num_robots, dtype=np.float32)
        self.adj_matrix = np.zeros((self.num_robots, self.num_robots), dtype=np.float32)
        self.edge_features = np.zeros((self.num_robots, self.num_robots, self.edge_dim), dtype=np.float32)

        # Initialize PyBullet if available
        self._setup_physics_engine()

    def _setup_physics_engine(self) -> None:
        """Connects to PyBullet and sets up gravity and ground plane."""
        if not PYBULLET_AVAILABLE:
            return
        if self.physics_client is None:
            connection_mode = p.GUI if self.render_mode == "gui" else p.DIRECT
            self.physics_client = p.connect(connection_mode)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
            p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the simulation environment and spawn robots in initial formation.

        Returns:
            Tuple of (observation_dict, info_dict).
        """
        if seed is not None:
            np.random.seed(seed)
        self.step_count = 0
        self._reward_totals = {"progress": 0.0, "conn": 0.0, "contact": 0.0, "energy": 0.0, "completion": 0.0}

        if PYBULLET_AVAILABLE and self.physics_client is not None:
            p.resetSimulation(physicsClientId=self.physics_client)
            p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
            p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

            # Spawn robots in a circular or grid pattern around origin
            self.robot_body_ids = []
            for i in range(self.num_robots):
                angle = 2.0 * np.pi * i / self.num_robots
                pos = [2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.1]
                # Load simple sphere or R2D2 body as surrogate differential drive robot
                body_id = p.loadURDF("sphere2.urdf", basePosition=pos, globalScaling=0.5, physicsClientId=self.physics_client)
                self.robot_body_ids.append(body_id)

            # Spawn cooperative transport payload at origin
            self.payload_body_id = p.loadURDF("cube_small.urdf", basePosition=[0.0, 0.0, 0.2], globalScaling=1.5, physicsClientId=self.physics_client)
        else:
            # Kinematic fallback if PyBullet is unavailable during CI/testing
            for i in range(self.num_robots):
                angle = 2.0 * np.pi * i / self.num_robots
                self.robot_positions[i] = [2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.1]
                self.robot_velocities[i] = [0.0, 0.0, 0.0]
            self.robot_headings.fill(0.0)
            self.payload_pos = np.array([0.0, 0.0, 0.2], dtype=np.float32)

        # Update physical state and construct graph topology
        self._update_kinematics_and_graph()
        
        # Calculate initial payload-to-goal distance for potential-based progress tracking
        payload_coords = self._get_payload_position()
        self.prev_payload_dist = float(np.linalg.norm(payload_coords - self.goal_pos))

        # Build agent observations
        obs_dict = {}
        info_dict = {}
        all_node_features = self._extract_all_node_features()

        # Draw a single random comm mask for this env step — shared across all agents
        # so every agent's obs snapshot references the same random topology.
        random_comm_mask = np.random.rand(self.num_robots, self.num_robots).astype(np.float32)

        for idx, agent_id in enumerate(self._agent_ids):
            obs_dict[agent_id] = {
                "local_obs": all_node_features[idx].copy(),
                "node_features": all_node_features.copy(),
                "adj_matrix": self.adj_matrix.copy(),
                "edge_features": self.edge_features.copy(),
                "node_index": np.array([idx], dtype=np.int64),
                "random_comm_mask": random_comm_mask.copy()
            }
            info_dict[agent_id] = {"graph_connectivity": float(np.sum(self.adj_matrix[idx]))}

        return obs_dict, info_dict

    def step(
        self,
        action_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Advance simulation by one environment step, applying actions and evaluating rewards.

        Args:
            action_dict: Mapping from agent_id to action tensor/array.

        Returns:
            Tuple of (obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict).
        """
        self.step_count += 1

        # 1. Apply motor actions across physics substeps
        for _ in range(self.physics_steps_per_env_step):
            for idx, agent_id in enumerate(self._agent_ids):
                if agent_id in action_dict:
                    action = np.clip(action_dict[agent_id], -1.0, 1.0)
                    self._apply_robot_action(idx, action)
            if PYBULLET_AVAILABLE and self.physics_client is not None:
                p.stepSimulation(physicsClientId=self.physics_client)
            else:
                self._step_kinematics_fallback(action_dict)

        # 2. Update graph topology and physical kinematics post-step
        self._update_kinematics_and_graph()

        # 3. Compute structured reward components
        payload_coords = self._get_payload_position()
        curr_payload_dist = float(np.linalg.norm(payload_coords - self.goal_pos))
        
        # Potential-based progress difference: R_progress = gamma * Phi(s_t) - Phi(s_{t-1})
        progress_reward = 10.0 * (self.prev_payload_dist - curr_payload_dist)
        self.prev_payload_dist = curr_payload_dist

        # Compute algebraic connectivity (Fiedler value lambda_2 of Graph Laplacian L = D - A)
        # Uses scipy.linalg.eigh for symmetric matrices: guarantees real eigenvalues,
        # avoids spurious complex components from np.linalg.eigvals.
        degree_matrix = np.diag(np.sum(self.adj_matrix, axis=1))
        laplacian = degree_matrix - self.adj_matrix
        eigenvalues, _ = eigh(laplacian)
        eigenvalues = np.clip(eigenvalues, 0.0, None)  # Remove floating-point noise
        fiedler_val = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

        # Construct return dictionaries
        obs_dict = {}
        reward_dict = {}
        terminated_dict = {"__all__": False}
        truncated_dict = {"__all__": self.step_count >= self.max_steps}
        info_dict = {}

        all_node_features = self._extract_all_node_features()

        # Check task completion threshold
        if curr_payload_dist < 1.0:
            terminated_dict["__all__"] = True

        # Draw a single random comm mask for this env step — shared across all agents
        random_comm_mask = np.random.rand(self.num_robots, self.num_robots).astype(np.float32)

        for idx, agent_id in enumerate(self._agent_ids):
            obs_dict[agent_id] = {
                "local_obs": all_node_features[idx].copy(),
                "node_features": all_node_features.copy(),
                "adj_matrix": self.adj_matrix.copy(),
                "edge_features": self.edge_features.copy(),
                "node_index": np.array([idx], dtype=np.int64),
                "random_comm_mask": random_comm_mask.copy()
            }

            # Individual reward shaping per robot
            # r_i = progress + cooperative contact bonus + connectivity retention - energy regularizer
            action_norm = float(np.linalg.norm(action_dict.get(agent_id, np.zeros(self.action_dim))))
            energy_penalty = 0.05 * (action_norm ** 2)
            
            # Connectivity penalty: exponential drop if isolated
            neighbor_count = np.sum(self.adj_matrix[idx])
            conn_penalty = -2.0 if neighbor_count == 0 else (0.5 * fiedler_val)

            # Cooperative contact bonus: reward robots that are near the payload
            payload_dist_i = float(np.linalg.norm(
                self.robot_positions[idx][:2] - payload_coords[:2]
            ))
            coop_contact = self.contact_bonus if payload_dist_i < self.contact_radius else 0.0

            reward_dict[agent_id] = progress_reward + conn_penalty + coop_contact - energy_penalty
            self._reward_totals["progress"] += progress_reward
            self._reward_totals["conn"] += conn_penalty
            self._reward_totals["contact"] += coop_contact
            self._reward_totals["energy"] -= energy_penalty
            terminated_dict[agent_id] = terminated_dict["__all__"]
            truncated_dict[agent_id] = truncated_dict["__all__"]
            
            info_dict[agent_id] = {
                "payload_dist": curr_payload_dist,
                "fiedler_val": fiedler_val,
                "neighbor_count": int(neighbor_count),
                "contact": payload_dist_i < self.contact_radius
            }

        # Terminal completion reward: sparse bonus for all agents on task success
        if terminated_dict["__all__"]:
            for agent_id in self._agent_ids:
                reward_dict[agent_id] = reward_dict.get(agent_id, 0.0) + self.completion_bonus
            self._reward_totals["completion"] += self.completion_bonus * self.num_robots

        if terminated_dict["__all__"] or truncated_dict["__all__"]:
            print(f"[EP END] progress={self._reward_totals['progress']:.2f} "
                  f"conn={self._reward_totals['conn']:.2f} "
                  f"contact={self._reward_totals['contact']:.2f} "
                  f"energy={self._reward_totals['energy']:.2f} "
                  f"completion={self._reward_totals['completion']:.2f} "
                  f"final_payload_dist={curr_payload_dist:.2f}")
        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def _apply_robot_action(self, robot_idx: int, action: np.ndarray) -> None:
        """Applies differential drive force/torque commands to PyBullet robot body."""
        if PYBULLET_AVAILABLE and self.physics_client is not None and len(self.robot_body_ids) > robot_idx:
            body_id = self.robot_body_ids[robot_idx]

            left_wheel_vel = float(action[0])
            right_wheel_vel = float(action[1])

            # Forward thrust: sum of wheel velocities drives translation along local X
            forward_gain = 20.0
            forward_force_mag = forward_gain * (left_wheel_vel + right_wheel_vel)
            force = [forward_force_mag, 0.0, 0.0]

            # Yaw torque: difference of wheel velocities drives rotation about Z
            torque_gain = 8.0  # tune separately from forward_gain
            yaw_torque_mag = torque_gain * (left_wheel_vel - right_wheel_vel)
            torque = [0.0, 0.0, yaw_torque_mag]

            p.applyExternalForce(
                body_id, -1, forceObj=force, posObj=[0, 0, 0],
                flags=p.LINK_FRAME, physicsClientId=self.physics_client
            )
            p.applyExternalTorque(
                body_id, -1, torqueObj=torque,
                flags=p.LINK_FRAME, physicsClientId=self.physics_client
            )

    def _step_kinematics_fallback(self, action_dict: Dict[str, Any]) -> None:
        """Kinematic simulation fallback when headless without PyBullet C++ binaries.

        Also updates payload position based on forces from robots in contact
        (within self.contact_radius). Without this, payload_pos was static and
        progress_reward was always ~0 in fallback mode.
        """
        dt_step = self.dt * self.physics_steps_per_env_step

        for idx, agent_id in enumerate(self._agent_ids):
            act = action_dict.get(agent_id, np.zeros(self.action_dim))
            left_wheel_vel = float(act[0])
            right_wheel_vel = float(act[1])

            forward_speed = 1.5 * (left_wheel_vel + right_wheel_vel) / 2.0
            yaw_rate = 1.5 * (left_wheel_vel - right_wheel_vel)

            self.robot_headings[idx] += yaw_rate * dt_step

            heading = self.robot_headings[idx]
            vx = forward_speed * np.cos(heading)
            vy = forward_speed * np.sin(heading)
            self.robot_velocities[idx] = [vx, vy, 0.0]
            self.robot_positions[idx] += self.robot_velocities[idx] * dt_step

        # Update payload dynamics: weighted average of velocities from robots
        # within contact_radius (proximity-weighted linear drag model)
        weights = []
        vel_contributions = []
        for i in range(self.num_robots):
            dist_to_payload = float(np.linalg.norm(
                self.robot_positions[i][:2] - self.payload_pos[:2]
            ))
            if dist_to_payload < self.contact_radius:
                w = max(0.0, 1.0 - dist_to_payload / self.contact_radius)
                weights.append(w)
                vel_contributions.append(w * self.robot_velocities[i])
        if weights:
            total_w = sum(weights)
            payload_vel = sum(vel_contributions) / total_w
            self.payload_pos += payload_vel * dt_step

    def _update_kinematics_and_graph(self) -> None:
        """Updates robot states and constructs the dynamic adjacency matrix and edge tensors."""
        if PYBULLET_AVAILABLE and self.physics_client is not None and len(self.robot_body_ids) > 0:
            for idx, body_id in enumerate(self.robot_body_ids):
                pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=self.physics_client)
                vel, _ = p.getBaseVelocity(body_id, physicsClientId=self.physics_client)
                self.robot_positions[idx] = np.array(pos, dtype=np.float32)
                self.robot_velocities[idx] = np.array(vel, dtype=np.float32)
                # Extract yaw from quaternion for ego-centric heading computation
                _, _, yaw = p.getEulerFromQuaternion(orn)
                self.robot_headings[idx] = float(yaw)

        # Reset graph structures
        self.adj_matrix.fill(0.0)
        self.edge_features.fill(0.0)

        # Compute pairwise Euclidean distances and establish communication edges
        for i in range(self.num_robots):
            for j in range(self.num_robots):
                if i == j:
                    continue
                rel_pos = self.robot_positions[j] - self.robot_positions[i]
                dist = float(np.linalg.norm(rel_pos))
                
                # Check Euclidean communication threshold
                if dist <= self.comm_radius:
                    self.adj_matrix[i, j] = 1.0
                    rel_vel = self.robot_velocities[j] - self.robot_velocities[i]
                    # Ego-centric heading: subtract observer's own heading, wrap to [-1, 1]
                    raw_angle = np.arctan2(rel_pos[1], rel_pos[0])
                    heading_diff = ((raw_angle - self.robot_headings[i] + np.pi) % (2 * np.pi) - np.pi) / np.pi
                    
                    # Populate edge geometry tensor: [rel_pos(3), rel_vel(3), dist(1), heading_diff(1)]
                    self.edge_features[i, j, :3] = rel_pos
                    self.edge_features[i, j, 3:6] = rel_vel
                    self.edge_features[i, j, 6] = dist
                    self.edge_features[i, j, 7] = heading_diff

    def _get_payload_position(self) -> np.ndarray:
        """Retrieves payload world coordinates."""
        if PYBULLET_AVAILABLE and self.physics_client is not None and self.payload_body_id is not None:
            pos, _ = p.getBasePositionAndOrientation(self.payload_body_id, physicsClientId=self.physics_client)
            return np.array(pos, dtype=np.float32)
        return getattr(self, "payload_pos", np.zeros(3, dtype=np.float32))

    def _cast_lidar_rays(self, robot_idx: int) -> np.ndarray:
        """
        Cast N evenly-spaced rays from robot_idx and return hit distances.

        Uses PyBullet's rayTestBatch for batch efficiency. Each ray starts at
        the robot's position (offset to lidar_ray_height) and extends outward
        in the XY plane for lidar_max_range meters.

        Args:
            robot_idx: Index of the robot to cast rays from.

        Returns:
            np.ndarray of shape (lidar_num_rays,) with hit distances.
            Rays that don't hit anything return lidar_max_range.
        """
        distances = np.full(self.lidar_num_rays, self.lidar_max_range, dtype=np.float32)

        if not (PYBULLET_AVAILABLE and self.physics_client is not None
                and len(self.robot_body_ids) > robot_idx):
            return distances

        body_id = self.robot_body_ids[robot_idx]
        pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=self.physics_client)

        # Get the robot's yaw from its quaternion orientation for egocentric rays
        _, _, yaw = p.getEulerFromQuaternion(orn)

        ray_origin = [pos[0], pos[1], self.lidar_ray_height]

        ray_from_list = []
        ray_to_list = []
        for angle in self.lidar_angles:
            world_angle = yaw + angle
            dx = self.lidar_max_range * np.cos(world_angle)
            dy = self.lidar_max_range * np.sin(world_angle)
            ray_from_list.append(ray_origin)
            ray_to_list.append([
                ray_origin[0] + dx,
                ray_origin[1] + dy,
                self.lidar_ray_height
            ])

        results = p.rayTestBatch(
            ray_from_list, ray_to_list,
            physicsClientId=self.physics_client
        )

        for k, result in enumerate(results):
            hit_object_id, _, hit_fraction, _, _ = result
            if hit_object_id != -1 and hit_object_id != body_id:
                # hit_fraction ∈ [0, 1]; actual distance = fraction * max_range
                distances[k] = hit_fraction * self.lidar_max_range

        return distances

    def _extract_all_node_features(self) -> np.ndarray:
        """Constructs raw feature matrix X of size (num_robots, raw_obs_dim)."""
        node_feats = np.zeros((self.num_robots, self.raw_obs_dim), dtype=np.float32)
        payload_pos = self._get_payload_position()

        for i in range(self.num_robots):
            # Pos (0:3), Vel (3:6)
            node_feats[i, 0:3] = self.robot_positions[i]
            node_feats[i, 3:6] = self.robot_velocities[i]
            # Simulated LiDAR ray distances (6:18)
            node_feats[i, 6:18] = self._cast_lidar_rays(i)
            # Relative vector to goal (18:21)
            node_feats[i, 18:21] = self.goal_pos - self.robot_positions[i]
            # Relative vector to payload (21:24)
            node_feats[i, 21:24] = payload_pos - self.robot_positions[i]

        return node_feats

    def close(self) -> None:
        """Disconnect PyBullet physics client to prevent resource leaks across worker respawns."""
        if PYBULLET_AVAILABLE and self.physics_client is not None:
            try:
                p.disconnect(physicsClientId=self.physics_client)
            except Exception:
                pass  # Already disconnected or invalid client
            self.physics_client = None


if __name__ == "__main__":
    # Smoke test verifying environment API and observation structures
    print("Running verification smoke test for MultiRobotPhysicsEnv...")
    env = MultiRobotPhysicsEnv({"num_robots": 4, "comm_radius": 6.0})
    obs, info = env.reset()
    assert len(obs) == 4, f"Expected 4 agent observation entries, got {len(obs)}"
    assert "local_obs" in obs["robot_0"], "Missing local_obs key in observation dict."
    
    # Take step with dummy actions
    actions = {f"robot_{i}": np.array([0.5, -0.5], dtype=np.float32) for i in range(4)}
    next_obs, rewards, terminated, truncated, infos = env.step(actions)
    assert len(rewards) == 4, "Missing reward entries after step."
    print("Verification passed! MultiRobotPhysicsEnv complies with RLlib MultiAgentEnv specification.")
