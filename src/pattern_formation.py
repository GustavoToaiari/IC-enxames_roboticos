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
    # Parâmetros do robô
    WHEEL_RADIUS = 0.0425 / 2
    AXLE_LENGTH = 0.054

    LEFT_SIGN = 1
    RIGHT_SIGN = 1

    MAX_WHEEL_SPEED = 10.0
    MAX_LINEAR_SPEED = 0.20
    MAX_ANGULAR_SPEED = 2.5

    LEADER_MAX_LINEAR_SPEED = 0.12
    FOLLOWER_MAX_LINEAR_SPEED = 0.20

    K_RHO = 1.5
    K_ALPHA = 3.0

    CONTROL_DT = 0.05
    YAW_OFFSET = math.pi / 2

    # Parâmetros da formação
    DESIRED_DISTANCE = 0.5
    POSITION_TOL = 0.05
    GOAL_TOL = 0.08
    DEBUG_PRINT_DT = 0.5
    WAIT_TIME = 2.0

    TRIANGLE_SIDE = -0.5

    # Estados
    STATE_ELECTION = "ELEICAO"
    STATE_FORM_LINE = "FORMANDO_LINHA"
    STATE_WAIT = "ESPERANDO"
    STATE_FORM_TRIANGLE = "FORMANDO_TRIANGULO"
    STATE_NAV_TRIANGLE = "NAVEGANDO_TRIANGULO"
    STATE_GOAL_REACHED = "GOAL_ALCANCADO"

    def __init__(self, sim):
        self.sim = sim
        self.state = self.STATE_ELECTION

        self.leader_key = None
        self.follower1_key = None
        self.follower2_key = None

        self.wait_start_time = None
        self.next_state_after_wait = None
        self.wait_reason = ""
        self.wait_end_message = ""

        self.last_debug_time = 0.0

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

        print("Handles obtidos com sucesso.")

    # Utilidades gerais
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

    # Controle de ponto
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
        position_tol=None
    ):
        x, y, yaw = poses[robot_key]
        v, w, rho, alpha, arrived = self.controller_to_point(
            x, y, yaw,
            x_goal, y_goal,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
            position_tol=position_tol
        )

        if arrived:
            self.stop_robot(robot_key)
        else:
            self.set_wheel_speeds(robot_key, v, w)

        return rho, alpha, arrived

    # Controle genérico de espera
    def start_wait(self, next_state, reason="", end_message=""):
        self.stop_all_robots()
        self.wait_start_time = time.time()
        self.next_state_after_wait = next_state
        self.wait_reason = reason
        self.wait_end_message = end_message
        self.state = self.STATE_WAIT

    def handle_wait(self):
        self.stop_all_robots()
        self.debug_wait_status()

        if self.wait_start_time is None:
            return

        if (time.time() - self.wait_start_time) >= self.WAIT_TIME:
            if self.wait_end_message:
                print(f"\n{self.wait_end_message}\n")

            self.state = self.next_state_after_wait
            self.wait_start_time = None
            self.next_state_after_wait = None
            self.wait_reason = ""
            self.wait_end_message = ""

    # Lógica do líder
    def elect_leader(self, poses):
        distances_to_origin = {}
        for key, (x, y, _) in poses.items():
            distances_to_origin[key] = math.hypot(x, y)

        self.leader_key = min(distances_to_origin, key=distances_to_origin.get)

        others = [key for key in self.robots if key != self.leader_key]

        xL, yL, _ = poses[self.leader_key]
        others.sort(key=lambda key: self.distance_2d(xL, yL, poses[key][0], poses[key][1]))

        self.follower1_key = others[0]
        self.follower2_key = others[1]

        print(f"\nLíder eleito: {self.robots[self.leader_key].name}")
        print(f"Seguidor 1: {self.robots[self.follower1_key].name}")
        print(f"Seguidor 2: {self.robots[self.follower2_key].name}\n")

    # Formação em linha
    def get_line_goals(self, poses):
        xL, yL, yawL = poses[self.leader_key]

        xF1_goal = xL - self.DESIRED_DISTANCE * math.cos(yawL)
        yF1_goal = yL - self.DESIRED_DISTANCE * math.sin(yawL)

        xF2_goal = xL - 2.0 * self.DESIRED_DISTANCE * math.cos(yawL)
        yF2_goal = yL - 2.0 * self.DESIRED_DISTANCE * math.sin(yawL)

        return (xF1_goal, yF1_goal), (xF2_goal, yF2_goal)

    def formacao_linha(self, poses):
        self.stop_robot(self.leader_key)

        (xF1_goal, yF1_goal), (xF2_goal, yF2_goal) = self.get_line_goals(poses)

        rho1, alpha1, arrived1 = self.go_to_point(
            self.follower1_key, xF1_goal, yF1_goal, poses,
            max_linear_speed=self.FOLLOWER_MAX_LINEAR_SPEED
        )
        rho2, alpha2, arrived2 = self.go_to_point(
            self.follower2_key, xF2_goal, yF2_goal, poses,
            max_linear_speed=self.FOLLOWER_MAX_LINEAR_SPEED
        )

        self.debug_line_status(
            poses,
            xF1_goal, yF1_goal,
            xF2_goal, yF2_goal,
            rho1, alpha1,
            rho2, alpha2,
            arrived1, arrived2
        )

        if arrived1 and arrived2:
            self.stop_robot(self.follower1_key)
            self.stop_robot(self.follower2_key)

            print("\nFormação em linha concluída.")
            print("Aguardando 2 segundos antes de iniciar o triângulo...\n")

            self.start_wait(
                next_state=self.STATE_FORM_TRIANGLE,
                reason="Pausa após formação em linha",
                end_message="Iniciando formação em triângulo..."
            )

    # Formação em triângulo parada
    # Base = líder + seguidor2
    # Quem sobe = seguidor1
    def get_triangle_goal_stationary(self, poses):
        xL, yL, _ = poses[self.leader_key]
        xB, yB, _ = poses[self.follower2_key]

        dx = xB - xL
        dy = yB - yL

        base_len = math.hypot(dx, dy)
        if base_len < 1e-6:
            raise RuntimeError("Base do triângulo muito pequena.")

        ux = dx / base_len
        uy = dy / base_len

        px = -uy
        py = ux

        mx = (xL + xB) / 2.0
        my = (yL + yB) / 2.0

        h = (math.sqrt(3) / 2.0) * base_len

        x_apex = mx + self.TRIANGLE_SIDE * h * px
        y_apex = my + self.TRIANGLE_SIDE * h * py

        return x_apex, y_apex, base_len

    def formacao_triangulo(self, poses):
        self.stop_robot(self.leader_key)
        self.stop_robot(self.follower2_key)

        x_goal, y_goal, base_len = self.get_triangle_goal_stationary(poses)

        rho, alpha, arrived = self.go_to_point(
            self.follower1_key, x_goal, y_goal, poses,
            max_linear_speed=self.FOLLOWER_MAX_LINEAR_SPEED
        )

        self.debug_triangle_status(
            poses,
            x_goal, y_goal,
            rho, alpha,
            arrived,
            base_len
        )

        if arrived:
            self.stop_robot(self.follower1_key)

            print("\nFormação em triângulo concluída.")
            print("Aguardando 2 segundos antes de iniciar a navegação...\n")

            self.start_wait(
                next_state=self.STATE_NAV_TRIANGLE,
                reason="Pausa após formação em triângulo",
                end_message="Iniciando navegação até goal1 mantendo a formação..."
            )

    # Navegação em triângulo
    def get_moving_triangle_goals(self, poses):
        xL, yL, yawL = poses[self.leader_key]
        d = self.DESIRED_DISTANCE

        xF2_goal, yF2_goal = self.local_to_world(
            xL, yL, yawL,
            -2.0 * d, 0.0
        )

        xF1_goal, yF1_goal = self.local_to_world(
            xL, yL, yawL,
            -1.0 * d,
            self.TRIANGLE_SIDE * math.sqrt(3) * d
        )

        return (xF1_goal, yF1_goal), (xF2_goal, yF2_goal)

    def navegar_em_triangulo(self, poses):
        goal_x, goal_y = self.get_goal_position()

        rhoL, alphaL, arrivedL = self.go_to_point(
            self.leader_key, goal_x, goal_y, poses,
            max_linear_speed=self.LEADER_MAX_LINEAR_SPEED,
            position_tol=self.GOAL_TOL
        )

        if arrivedL:
            self.stop_robot(self.leader_key)

        (xF1_goal, yF1_goal), (xF2_goal, yF2_goal) = self.get_moving_triangle_goals(poses)

        rho1, alpha1, arrived1 = self.go_to_point(
            self.follower1_key, xF1_goal, yF1_goal, poses,
            max_linear_speed=self.FOLLOWER_MAX_LINEAR_SPEED
        )
        rho2, alpha2, arrived2 = self.go_to_point(
            self.follower2_key, xF2_goal, yF2_goal, poses,
            max_linear_speed=self.FOLLOWER_MAX_LINEAR_SPEED
        )

        self.debug_nav_triangle_status(
            poses,
            goal_x, goal_y,
            xF1_goal, yF1_goal,
            xF2_goal, yF2_goal,
            rhoL, alphaL, arrivedL,
            rho1, alpha1, arrived1,
            rho2, alpha2, arrived2
        )

        if arrivedL and arrived1 and arrived2:
            self.stop_all_robots()
            self.state = self.STATE_GOAL_REACHED
            print("\nGoal alcançado com a formação triangular mantida.\n")

    # Debug
    def debug_line_status(
        self,
        poses,
        xF1_goal, yF1_goal,
        xF2_goal, yF2_goal,
        rho1, alpha1,
        rho2, alpha2,
        arrived1, arrived2
    ):
        now = time.time()
        if now - self.last_debug_time < self.DEBUG_PRINT_DT:
            return

        self.last_debug_time = now

        xL, yL, yawL = poses[self.leader_key]
        x1, y1, _ = poses[self.follower1_key]
        x2, y2, _ = poses[self.follower2_key]

        print(
            f"[{self.state}] "
            f"Líder={self.robots[self.leader_key].name} "
            f"pos=({xL:.2f}, {yL:.2f}) yaw={yawL:.2f} | "
            f"{self.robots[self.follower1_key].name}: pos=({x1:.2f}, {y1:.2f}) "
            f"goal=({xF1_goal:.2f}, {yF1_goal:.2f}) rho={rho1:.3f} alpha={alpha1:.3f} arrived={arrived1} | "
            f"{self.robots[self.follower2_key].name}: pos=({x2:.2f}, {y2:.2f}) "
            f"goal=({xF2_goal:.2f}, {yF2_goal:.2f}) rho={rho2:.3f} alpha={alpha2:.3f} arrived={arrived2}"
        )

    def debug_wait_status(self):
        now = time.time()
        if now - self.last_debug_time < self.DEBUG_PRINT_DT:
            return

        self.last_debug_time = now

        if self.wait_start_time is None:
            return

        elapsed = now - self.wait_start_time
        remaining = max(0.0, self.WAIT_TIME - elapsed)

        print(
            f"[{self.state}] "
            f"{self.wait_reason} | faltam {remaining:.2f} s"
        )

    def debug_triangle_status(
        self,
        poses,
        x_goal, y_goal,
        rho, alpha,
        arrived,
        base_len
    ):
        now = time.time()
        if now - self.last_debug_time < self.DEBUG_PRINT_DT:
            return

        self.last_debug_time = now

        xL, yL, _ = poses[self.leader_key]
        x1, y1, _ = poses[self.follower1_key]
        x2, y2, _ = poses[self.follower2_key]

        print(
            f"[{self.state}] "
            f"Base: {self.robots[self.leader_key].name}=({xL:.2f}, {yL:.2f}) | "
            f"{self.robots[self.follower2_key].name}=({x2:.2f}, {y2:.2f}) | "
            f"base_len={base_len:.3f} | "
            f"{self.robots[self.follower1_key].name}: pos=({x1:.2f}, {y1:.2f}) "
            f"goal=({x_goal:.2f}, {y_goal:.2f}) rho={rho:.3f} alpha={alpha:.3f} arrived={arrived}"
        )

    def debug_nav_triangle_status(
        self,
        poses,
        goal_x, goal_y,
        xF1_goal, yF1_goal,
        xF2_goal, yF2_goal,
        rhoL, alphaL, arrivedL,
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
            f"[{self.state}] "
            f"goal1=({goal_x:.2f}, {goal_y:.2f}) | "
            f"Líder={self.robots[self.leader_key].name}: pos=({xL:.2f}, {yL:.2f}) yaw={yawL:.2f} "
            f"rho={rhoL:.3f} alpha={alphaL:.3f} arrived={arrivedL} | "
            f"{self.robots[self.follower1_key].name}: pos=({x1:.2f}, {y1:.2f}) "
            f"goal=({xF1_goal:.2f}, {yF1_goal:.2f}) rho={rho1:.3f} alpha={alpha1:.3f} arrived={arrived1} | "
            f"{self.robots[self.follower2_key].name}: pos=({x2:.2f}, {y2:.2f}) "
            f"goal=({xF2_goal:.2f}, {yF2_goal:.2f}) rho={rho2:.3f} alpha={alpha2:.3f} arrived={arrived2}"
        )

    # Máquina de estados
    def step(self):
        poses = self.get_all_poses()

        if self.state == self.STATE_ELECTION:
            self.elect_leader(poses)
            self.state = self.STATE_FORM_LINE

        elif self.state == self.STATE_FORM_LINE:
            self.formacao_linha(poses)

        elif self.state == self.STATE_WAIT:
            self.handle_wait()

        elif self.state == self.STATE_FORM_TRIANGLE:
            self.formacao_triangulo(poses)

        elif self.state == self.STATE_NAV_TRIANGLE:
            self.navegar_em_triangulo(poses)

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