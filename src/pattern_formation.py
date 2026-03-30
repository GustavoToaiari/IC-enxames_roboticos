import math
import time
from dataclasses import dataclass
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


@dataclass
class Robot:
    name: str
    base: int
    left: int
    right: int


class RobotFormation:
    # =========================
    # Parâmetros do robô
    # =========================
    WHEEL_RADIUS = 0.0425 / 2
    AXLE_LENGTH = 0.054

    LEFT_SIGN = 1
    RIGHT_SIGN = 1

    MAX_WHEEL_SPEED = 10.0
    MAX_LINEAR_SPEED = 0.20
    MAX_ANGULAR_SPEED = 2.5

    LEADER_MAX_LINEAR_SPEED = 0.12
    FOLLOWER_MAX_LINEAR_SPEED = 0.20
    LEADER_LINE_MAX_LINEAR_SPEED = 0.10
    FOLLOWER_LINE_MAX_LINEAR_SPEED = 0.18

    K_RHO = 1.5
    K_ALPHA = 3.0

    CONTROL_DT = 0.05
    YAW_OFFSET = math.pi / 2

    # =========================
    # Parâmetros da formação
    # =========================
    POSITION_TOL = 0.05
    GOAL_TOL = 0.08
    DEBUG_PRINT_DT = 0.5

    # triângulo equilátero com lado = 2*d
    DESIRED_DISTANCE = 0.25

    # linha atrás do líder
    LINE_SPACING = 0.50

    # +1 ou -1 define o lado do ápice
    TRIANGLE_SIDE = -1.0

    # =========================
    # Peer detection
    # =========================
    PEER_RANGE = 2.0
    PEER_FOV_DEG = 360.0

    # =========================
    # Campo potencial entre robôs
    # =========================
    ENABLE_INTER_ROBOT_AVOIDANCE = True
    ENABLE_ENV_OBSTACLE_AVOIDANCE_FOR_LEADER = True

    ROBOT_RADIUS = 0.0724 / 2
    RHO_0 = 0.20
    K_ATT = 1.00
    K_REP = 0.03
    K_ROT = 0.15
    MIN_DISTANCE_EPS = 1e-4

    # distância de segurança centro-centro entre dois robôs
    R_CLEAR_RR = 2.0 * ROBOT_RADIUS

    # =========================
    # Obstáculos do ambiente
    # =========================
    OBSTACLE_RADIUS = 0.50  # 50 cm de raio
    LINE_ENTER_DSURF = 1.00
    LINE_EXIT_DSURF = 1.30
    CLEAR_HOLD_TIME = 1.0

    # Estados
    STATE_PEER_DETECTION = "PEER_DETECTION"
    STATE_FORM_TRIANGLE = "FORM_TRIANGLE"
    STATE_NAV_TRIANGLE = "NAV_TRIANGLE"
    STATE_FORM_LINE = "FORM_LINE"
    STATE_NAV_LINE = "NAV_LINE"
    STATE_GOAL_REACHED = "GOAL_REACHED"

    def __init__(self, sim):
        self.sim = sim
        self.state = self.STATE_PEER_DETECTION

        self.leader_key = None
        self.follower1_key = None
        self.follower2_key = None

        self.last_debug_time = 0.0
        self.clear_obstacle_start_time = None

        self.goal_handle = self._get_required_object("/goal1")

        self.robots = {
            "r1": Robot(
                name="ePuck1",
                base=self._get_required_object("/base1"),
                left=self._get_required_object("/leftJoint1"),
                right=self._get_required_object("/rightJoint1"),
            ),
            "r2": Robot(
                name="ePuck2",
                base=self._get_required_object("/base2"),
                left=self._get_required_object("/leftJoint2"),
                right=self._get_required_object("/rightJoint2"),
            ),
            "r3": Robot(
                name="ePuck3",
                base=self._get_required_object("/base3"),
                left=self._get_required_object("/leftJoint3"),
                right=self._get_required_object("/rightJoint3"),
            ),
        }

        self.obstacles = [
            {"name": "Obstacle1", "handle": self._get_required_object("/Obstacle1"), "radius": self.OBSTACLE_RADIUS},
            {"name": "Obstacle2", "handle": self._get_required_object("/Obstacle2"), "radius": self.OBSTACLE_RADIUS},
        ]

        print("Handles obtidos com sucesso.")
        print("Obstáculos carregados:")
        for obs in self.obstacles:
            print(f" - {obs['name']} | raio={obs['radius']:.2f} m")

    # =========================
    # Utilidades gerais
    # =========================
    def _get_required_object(self, path):
        handle = self.sim.getObject(path)
        if handle == -1:
            raise RuntimeError(f"Objeto não encontrado no CoppeliaSim: {path}")
        return handle

    def wrap_to_pi(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def get_pose_2d(self, base_handle):
        pos = self.sim.getObjectPosition(base_handle, -1)
        ori = self.sim.getObjectOrientation(base_handle, -1)

        x = pos[0]
        y = pos[1]
        yaw = self.wrap_to_pi(ori[2] + self.YAW_OFFSET)

        return x, y, yaw

    def get_all_poses(self):
        poses = {}
        for key, robot in self.robots.items():
            poses[key] = self.get_pose_2d(robot.base)
        return poses

    def get_goal_position(self):
        pos = self.sim.getObjectPosition(self.goal_handle, -1)
        return pos[0], pos[1]

    def get_obstacle_positions(self):
        obstacles = []
        for obs in self.obstacles:
            pos = self.sim.getObjectPosition(obs["handle"], -1)
            obstacles.append((obs["name"], pos[0], pos[1], obs["radius"]))
        return obstacles

    def set_wheel_speeds(self, robot_key, v, w):
        robot = self.robots[robot_key]

        wl = (v - (self.AXLE_LENGTH / 2.0) * w) / self.WHEEL_RADIUS
        wr = (v + (self.AXLE_LENGTH / 2.0) * w) / self.WHEEL_RADIUS

        wl *= self.LEFT_SIGN
        wr *= self.RIGHT_SIGN

        wl = max(-self.MAX_WHEEL_SPEED, min(self.MAX_WHEEL_SPEED, wl))
        wr = max(-self.MAX_WHEEL_SPEED, min(self.MAX_WHEEL_SPEED, wr))

        self.sim.setJointTargetVelocity(robot.left, wl)
        self.sim.setJointTargetVelocity(robot.right, wr)

    def stop_robot(self, robot_key):
        robot = self.robots[robot_key]
        self.sim.setJointTargetVelocity(robot.left, 0.0)
        self.sim.setJointTargetVelocity(robot.right, 0.0)

    def stop_all_robots(self):
        for key in self.robots:
            self.stop_robot(key)

    def distance_2d(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def local_to_world(self, x_ref, y_ref, yaw_ref, dx_local, dy_local):
        x_world = x_ref + dx_local * math.cos(yaw_ref) - dy_local * math.sin(yaw_ref)
        y_world = y_ref + dx_local * math.sin(yaw_ref) + dy_local * math.cos(yaw_ref)
        return x_world, y_world

    # =========================
    # Campo repulsivo + rotacional
    # =========================
    def repulsive_rotational_surface(self, px, py, ox, oy, gx, gy, r_clear):
        dx = px - ox
        dy = py - oy
        d = math.hypot(dx, dy)

        if d < self.MIN_DISTANCE_EPS:
            d = self.MIN_DISTANCE_EPS

        d_surf = d - r_clear
        d_surf_safe = max(d_surf, self.MIN_DISTANCE_EPS)

        ex = dx / d
        ey = dy / d

        ogx = gx - ox
        ogy = gy - oy

        cross = ex * ogy - ey * ogx
        cross_sign = 1.0 if cross >= 0.0 else -1.0

        if d_surf >= self.RHO_0:
            return 0.0, 0.0, d_surf

        mag = (1.0 / d_surf_safe - 1.0 / self.RHO_0) * (1.0 / (d_surf_safe * d_surf_safe))

        mag_rep = self.K_REP * mag
        fx_rep = mag_rep * ex
        fy_rep = mag_rep * ey

        mag_rot = self.K_ROT * mag
        tx = -ey * cross_sign
        ty = ex * cross_sign
        fx_rot = mag_rot * tx
        fy_rot = mag_rot * ty

        return fx_rep + fx_rot, fy_rep + fy_rot, d_surf

    # =========================
    # Campo entre robôs
    # =========================
    def compute_inter_robot_field(self, robot_key, poses, x_goal, y_goal):
        x, y, _ = poses[robot_key]

        fx = self.K_ATT * (x_goal - x)
        fy = self.K_ATT * (y_goal - y)

        if not self.ENABLE_INTER_ROBOT_AVOIDANCE:
            return fx, fy

        for other_key, other_pose in poses.items():
            if other_key == robot_key:
                continue

            ox, oy, _ = other_pose
            fxi, fyi, _ = self.repulsive_rotational_surface(
                x, y,
                ox, oy,
                x_goal, y_goal,
                self.R_CLEAR_RR
            )
            fx += fxi
            fy += fyi

        return fx, fy

    # =========================
    # Campo dos obstáculos do ambiente
    # =========================
    def compute_env_obstacle_field(self, x, y, x_goal, y_goal):
        fx = 0.0
        fy = 0.0
        min_d_surf = float("inf")

        for _, ox, oy, obs_radius in self.get_obstacle_positions():
            fxi, fyi, d_surf = self.repulsive_rotational_surface(
                x, y,
                ox, oy,
                x_goal, y_goal,
                obs_radius + self.ROBOT_RADIUS
            )
            fx += fxi
            fy += fyi
            min_d_surf = min(min_d_surf, d_surf)

        return fx, fy, min_d_surf

    def get_leader_min_obstacle_surface_distance(self, poses):
        xL, yL, _ = poses[self.leader_key]
        goal_x, goal_y = self.get_goal_position()
        _, _, min_d_surf = self.compute_env_obstacle_field(xL, yL, goal_x, goal_y)
        return min_d_surf

    # =========================
    # Controle de ponto
    # =========================
    def controller_to_point(
        self,
        x, y, yaw,
        x_goal, y_goal,
        max_linear_speed=None,
        max_angular_speed=None,
        position_tol=None
    ):
        if max_linear_speed is None:
            max_linear_speed = self.MAX_LINEAR_SPEED
        if max_angular_speed is None:
            max_angular_speed = self.MAX_ANGULAR_SPEED
        if position_tol is None:
            position_tol = self.POSITION_TOL

        dx = x_goal - x
        dy = y_goal - y

        rho = math.hypot(dx, dy)
        desired_theta = math.atan2(dy, dx)
        alpha = self.wrap_to_pi(desired_theta - yaw)

        if rho < position_tol:
            return 0.0, 0.0, rho, alpha, True

        if abs(alpha) > math.pi / 2:
            v = 0.0
        else:
            v = self.K_RHO * rho * max(0.0, math.cos(alpha))

        w = self.K_ALPHA * alpha

        v = max(-max_linear_speed, min(max_linear_speed, v))
        w = max(-max_angular_speed, min(max_angular_speed, w))

        return v, w, rho, alpha, False

    def go_to_point(
        self,
        robot_key,
        x_goal, y_goal,
        poses,
        max_linear_speed=None,
        max_angular_speed=None,
        position_tol=None,
        use_env_obstacles=False
    ):
        x, y, yaw = poses[robot_key]

        if max_linear_speed is None:
            max_linear_speed = self.MAX_LINEAR_SPEED
        if max_angular_speed is None:
            max_angular_speed = self.MAX_ANGULAR_SPEED
        if position_tol is None:
            position_tol = self.POSITION_TOL

        dx_goal = x_goal - x
        dy_goal = y_goal - y
        rho_real = math.hypot(dx_goal, dy_goal)

        if rho_real < position_tol:
            self.stop_robot(robot_key)
            alpha_real = self.wrap_to_pi(math.atan2(dy_goal, dx_goal) - yaw) if rho_real > self.MIN_DISTANCE_EPS else 0.0
            return rho_real, alpha_real, True

        fx, fy = self.compute_inter_robot_field(robot_key, poses, x_goal, y_goal)

        if use_env_obstacles and self.ENABLE_ENV_OBSTACLE_AVOIDANCE_FOR_LEADER:
            fx_env, fy_env, _ = self.compute_env_obstacle_field(x, y, x_goal, y_goal)
            fx += fx_env
            fy += fy_env

        if math.hypot(fx, fy) < self.MIN_DISTANCE_EPS:
            x_virtual = x_goal
            y_virtual = y_goal
        else:
            x_virtual = x + fx
            y_virtual = y + fy

        v, w, _, _, _ = self.controller_to_point(
            x, y, yaw,
            x_virtual, y_virtual,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
            position_tol=position_tol
        )

        self.set_wheel_speeds(robot_key, v, w)

        alpha_real = self.wrap_to_pi(math.atan2(dy_goal, dx_goal) - yaw)
        return rho_real, alpha_real, False

    # =========================
    # Peer detection
    # =========================
    def is_peer_detected(self, observer_pose, target_pose):
        x_obs, y_obs, yaw_obs = observer_pose
        x_t, y_t, _ = target_pose

        dx = x_t - x_obs
        dy = y_t - y_obs
        dist = math.hypot(dx, dy)

        if dist > self.PEER_RANGE:
            return False

        angle_to_target = math.atan2(dy, dx)
        rel_angle = self.wrap_to_pi(angle_to_target - yaw_obs)

        half_fov = math.radians(self.PEER_FOV_DEG / 2.0)
        return abs(rel_angle) <= half_fov

    def count_detected_peers(self, robot_key, poses):
        count = 0
        observer_pose = poses[robot_key]

        for other_key, other_pose in poses.items():
            if other_key == robot_key:
                continue
            if self.is_peer_detected(observer_pose, other_pose):
                count += 1

        return count

    # =========================
    # Eleição do líder
    # =========================
    def elect_leader(self, poses):
        goal_x, goal_y = self.get_goal_position()

        peer_count = {}
        goal_distance = {}

        for key, (x, y, _) in poses.items():
            peer_count[key] = self.count_detected_peers(key, poses)
            goal_distance[key] = self.distance_2d(x, y, goal_x, goal_y)

        self.leader_key = min(
            self.robots.keys(),
            key=lambda k: (-peer_count[k], goal_distance[k])
        )

        others = [key for key in self.robots if key != self.leader_key]

        xL, yL, _ = poses[self.leader_key]
        others.sort(key=lambda key: self.distance_2d(xL, yL, poses[key][0], poses[key][1]))

        self.follower1_key = others[0]
        self.follower2_key = others[1]

        print("\n=== ELEIÇÃO DO LÍDER ===")
        for key in self.robots:
            print(
                f"{self.robots[key].name}: peers={peer_count[key]} | "
                f"dist_goal={goal_distance[key]:.3f}"
            )

        print(f"\nLíder eleito: {self.robots[self.leader_key].name}")
        print(f"Seguidor 1: {self.robots[self.follower1_key].name}")
        print(f"Seguidor 2: {self.robots[self.follower2_key].name}\n")

    # =========================
    # Geometria do triângulo / linha
    # =========================
    def get_triangle_goals_relative_to_leader(self, poses):
        xL, yL, yawL = poses[self.leader_key]
        d = self.DESIRED_DISTANCE

        xF2_goal, yF2_goal = self.local_to_world(xL, yL, yawL, -2.0 * d, 0.0)
        xF1_goal, yF1_goal = self.local_to_world(
            xL, yL, yawL,
            -1.0 * d,
            self.TRIANGLE_SIDE * math.sqrt(3) * d
        )

        return (xF1_goal, yF1_goal), (xF2_goal, yF2_goal)

    def get_line_goals_relative_to_leader(self, poses):
        xL, yL, yawL = poses[self.leader_key]
        s = self.LINE_SPACING

        xF1_goal, yF1_goal = self.local_to_world(xL, yL, yawL, -1.0 * s, 0.0)
        xF2_goal, yF2_goal = self.local_to_world(xL, yL, yawL, -2.0 * s, 0.0)

        return (xF1_goal, yF1_goal), (xF2_goal, yF2_goal)

    # =========================
    # Formação inicial do triângulo
    # =========================
    def form_triangle(self, poses):
        self.stop_robot(self.leader_key)

        (xF1_goal, yF1_goal), (xF2_goal, yF2_goal) = self.get_triangle_goals_relative_to_leader(poses)

        rho1, alpha1, arrived1 = self.go_to_point(
            self.follower1_key,
            xF1_goal, yF1_goal,
            poses,
            max_linear_speed=self.FOLLOWER_MAX_LINEAR_SPEED
        )

        rho2, alpha2, arrived2 = self.go_to_point(
            self.follower2_key,
            xF2_goal, yF2_goal,
            poses,
            max_linear_speed=self.FOLLOWER_MAX_LINEAR_SPEED
        )

        self.debug_followers_status(
            poses, "FORM_TRIANGLE",
            xF1_goal, yF1_goal,
            xF2_goal, yF2_goal,
            rho1, alpha1, arrived1,
            rho2, alpha2, arrived2
        )

        if arrived1 and arrived2:
            self.stop_robot(self.follower1_key)
            self.stop_robot(self.follower2_key)
            self.state = self.STATE_NAV_TRIANGLE
            self.clear_obstacle_start_time = None
            print("\nFormação triangular concluída.")
            print("Iniciando navegação até goal1 mantendo a formação...\n")

    def form_line(self, poses):
        self.stop_robot(self.leader_key)

        (xF1_goal, yF1_goal), (xF2_goal, yF2_goal) = self.get_line_goals_relative_to_leader(poses)

        rho1, alpha1, arrived1 = self.go_to_point(
            self.follower1_key,
            xF1_goal, yF1_goal,
            poses,
            max_linear_speed=self.FOLLOWER_LINE_MAX_LINEAR_SPEED
        )

        rho2, alpha2, arrived2 = self.go_to_point(
            self.follower2_key,
            xF2_goal, yF2_goal,
            poses,
            max_linear_speed=self.FOLLOWER_LINE_MAX_LINEAR_SPEED
        )

        self.debug_followers_status(
            poses, "FORM_LINE",
            xF1_goal, yF1_goal,
            xF2_goal, yF2_goal,
            rho1, alpha1, arrived1,
            rho2, alpha2, arrived2
        )

        if arrived1 and arrived2:
            self.stop_robot(self.follower1_key)
            self.stop_robot(self.follower2_key)
            self.state = self.STATE_NAV_LINE
            self.clear_obstacle_start_time = None
            print("\nFormação em linha concluída.")
            print("Iniciando navegação até goal1 em linha...\n")

    # =========================
    # Navegação
    # =========================
    def navigate_triangle(self, poses):
        min_d_surf = self.get_leader_min_obstacle_surface_distance(poses)

        if min_d_surf < self.LINE_ENTER_DSURF:
            print(
                f"\nLíder sentiu obstáculo (d_surf={min_d_surf:.3f} m). "
                f"Mudando de triângulo para linha.\n"
            )
            self.state = self.STATE_FORM_LINE
            return

        goal_x, goal_y = self.get_goal_position()

        rhoL, alphaL, arrivedL = self.go_to_point(
            self.leader_key,
            goal_x, goal_y,
            poses,
            max_linear_speed=self.LEADER_MAX_LINEAR_SPEED,
            position_tol=self.GOAL_TOL,
            use_env_obstacles=False
        )

        if arrivedL:
            self.stop_robot(self.leader_key)

        (xF1_goal, yF1_goal), (xF2_goal, yF2_goal) = self.get_triangle_goals_relative_to_leader(poses)

        rho1, alpha1, arrived1 = self.go_to_point(
            self.follower1_key,
            xF1_goal, yF1_goal,
            poses,
            max_linear_speed=self.FOLLOWER_MAX_LINEAR_SPEED
        )

        rho2, alpha2, arrived2 = self.go_to_point(
            self.follower2_key,
            xF2_goal, yF2_goal,
            poses,
            max_linear_speed=self.FOLLOWER_MAX_LINEAR_SPEED
        )

        self.debug_nav_status(
            poses, "NAV_TRIANGLE",
            goal_x, goal_y,
            xF1_goal, yF1_goal,
            xF2_goal, yF2_goal,
            rhoL, alphaL, arrivedL,
            rho1, alpha1, arrived1,
            rho2, alpha2, arrived2,
            min_d_surf
        )

        if arrivedL and arrived1 and arrived2:
            self.stop_all_robots()
            self.state = self.STATE_GOAL_REACHED
            print("\nGoal alcançado com a formação triangular mantida.\n")

    def navigate_line(self, poses):
        min_d_surf = self.get_leader_min_obstacle_surface_distance(poses)
        now = time.time()

        if min_d_surf > self.LINE_EXIT_DSURF:
            if self.clear_obstacle_start_time is None:
                self.clear_obstacle_start_time = now
            elif now - self.clear_obstacle_start_time >= self.CLEAR_HOLD_TIME:
                print(
                    f"\nLíder saiu da região crítica (d_surf={min_d_surf:.3f} m). "
                    f"Voltando de linha para triângulo.\n"
                )
                self.state = self.STATE_FORM_TRIANGLE
                self.clear_obstacle_start_time = None
                return
        else:
            self.clear_obstacle_start_time = None

        goal_x, goal_y = self.get_goal_position()

        rhoL, alphaL, arrivedL = self.go_to_point(
            self.leader_key,
            goal_x, goal_y,
            poses,
            max_linear_speed=self.LEADER_LINE_MAX_LINEAR_SPEED,
            position_tol=self.GOAL_TOL,
            use_env_obstacles=True
        )

        if arrivedL:
            self.stop_robot(self.leader_key)

        (xF1_goal, yF1_goal), (xF2_goal, yF2_goal) = self.get_line_goals_relative_to_leader(poses)

        rho1, alpha1, arrived1 = self.go_to_point(
            self.follower1_key,
            xF1_goal, yF1_goal,
            poses,
            max_linear_speed=self.FOLLOWER_LINE_MAX_LINEAR_SPEED
        )

        rho2, alpha2, arrived2 = self.go_to_point(
            self.follower2_key,
            xF2_goal, yF2_goal,
            poses,
            max_linear_speed=self.FOLLOWER_LINE_MAX_LINEAR_SPEED
        )

        self.debug_nav_status(
            poses, "NAV_LINE",
            goal_x, goal_y,
            xF1_goal, yF1_goal,
            xF2_goal, yF2_goal,
            rhoL, alphaL, arrivedL,
            rho1, alpha1, arrived1,
            rho2, alpha2, arrived2,
            min_d_surf
        )

        if arrivedL and arrived1 and arrived2:
            self.stop_all_robots()
            self.state = self.STATE_GOAL_REACHED
            print("\nGoal alcançado com a formação em linha mantida.\n")

    # =========================
    # Debug
    # =========================
    def debug_followers_status(
        self,
        poses, label,
        xF1_goal, yF1_goal,
        xF2_goal, yF2_goal,
        rho1, alpha1, arrived1,
        rho2, alpha2, arrived2
    ):
        now = time.time()
        if now - self.last_debug_time < self.DEBUG_PRINT_DT:
            return

        self.last_debug_time = now

        xL, yL, yawL = poses[self.leader_key]
        x1, y1, _ = poses[self.follower1_key]
        x2, y2, _ = poses[self.follower2_key]

        print(
            f"[{label}] "
            f"Líder={self.robots[self.leader_key].name}: pos=({xL:.2f}, {yL:.2f}) yaw={yawL:.2f} | "
            f"{self.robots[self.follower1_key].name}: pos=({x1:.2f}, {y1:.2f}) "
            f"goal=({xF1_goal:.2f}, {yF1_goal:.2f}) rho={rho1:.3f} alpha={alpha1:.3f} arrived={arrived1} | "
            f"{self.robots[self.follower2_key].name}: pos=({x2:.2f}, {y2:.2f}) "
            f"goal=({xF2_goal:.2f}, {yF2_goal:.2f}) rho={rho2:.3f} alpha={alpha2:.3f} arrived={arrived2}"
        )

    def debug_nav_status(
        self,
        poses, label,
        goal_x, goal_y,
        xF1_goal, yF1_goal,
        xF2_goal, yF2_goal,
        rhoL, alphaL, arrivedL,
        rho1, alpha1, arrived1,
        rho2, alpha2, arrived2,
        min_d_surf
    ):
        now = time.time()
        if now - self.last_debug_time < self.DEBUG_PRINT_DT:
            return

        self.last_debug_time = now

        xL, yL, yawL = poses[self.leader_key]
        x1, y1, _ = poses[self.follower1_key]
        x2, y2, _ = poses[self.follower2_key]

        print(
            f"[{label}] "
            f"goal1=({goal_x:.2f}, {goal_y:.2f}) | d_surf_obs_lider={min_d_surf:.3f} | "
            f"Líder={self.robots[self.leader_key].name}: pos=({xL:.2f}, {yL:.2f}) yaw={yawL:.2f} "
            f"rho={rhoL:.3f} alpha={alphaL:.3f} arrived={arrivedL} | "
            f"{self.robots[self.follower1_key].name}: pos=({x1:.2f}, {y1:.2f}) "
            f"goal=({xF1_goal:.2f}, {yF1_goal:.2f}) rho={rho1:.3f} alpha={alpha1:.3f} arrived={arrived1} | "
            f"{self.robots[self.follower2_key].name}: pos=({x2:.2f}, {y2:.2f}) "
            f"goal=({xF2_goal:.2f}, {yF2_goal:.2f}) rho={rho2:.3f} alpha={alpha2:.3f} arrived={arrived2}"
        )

    # =========================
    # Máquina de estados
    # =========================
    def step(self):
        poses = self.get_all_poses()

        if self.state == self.STATE_PEER_DETECTION:
            self.elect_leader(poses)
            self.state = self.STATE_FORM_TRIANGLE

        elif self.state == self.STATE_FORM_TRIANGLE:
            self.form_triangle(poses)

        elif self.state == self.STATE_NAV_TRIANGLE:
            self.navigate_triangle(poses)

        elif self.state == self.STATE_FORM_LINE:
            self.form_line(poses)

        elif self.state == self.STATE_NAV_LINE:
            self.navigate_line(poses)

        elif self.state == self.STATE_GOAL_REACHED:
            self.stop_all_robots()

    def controle_formacao(self):
        while True:
            self.step()
            time.sleep(self.CONTROL_DT)


def main():
    client = RemoteAPIClient()
    sim = None
    robot_formation = None

    try:
        sim = client.getObject("sim")
        print("Conexão com o CoppeliaSim bem-sucedida!")

        sim.startSimulation()
        print("Simulação iniciada!")

        robot_formation = RobotFormation(sim)
        robot_formation.controle_formacao()

    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.")

    except Exception as e:
        print(f"\nErro durante a execução: {e}")

    finally:
        if robot_formation is not None:
            try:
                robot_formation.stop_all_robots()
            except:
                pass

        if sim is not None:
            try:
                sim.stopSimulation()
                print("Simulação finalizada.")
            except:
                pass


if __name__ == "__main__":
    main()