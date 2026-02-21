import math
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def yaw_from_quaternion(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw*qz + qx*qy), 1.0 - 2.0 * (qy*qy + qz*qz))

def saturate(x, a, b):
    return max(min(x, b), a)

def world_to_body(vx_w, vy_w, yaw):
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    vx_b = c * vx_w - s * vy_w
    vy_b = s * vx_w + c * vy_w
    return vx_b, vy_b

def vw_to_wheels(v, w, r, L):
    wr = (2.0*v + w*L) / (2.0*r)
    wl = (2.0*v - w*L) / (2.0*r)
    return wl, wr


def repulsive_rotational_surface(px, py, ox, oy, gx, gy, r_clear, rho0, k_rep, k_rot):
    dx = px - ox
    dy = py - oy
    d  = math.hypot(dx, dy)
    if d < 1e-6:
        d = 1e-6

    d_surf = d - r_clear
    ex = dx / d
    ey = dy / d

    # vetor obstáculo -> goal (pra escolher lado de contorno)
    ogx = gx - ox
    ogy = gy - oy
    cross = ex*ogy - ey*ogx
    cross_sign = 1.0 if cross >= 0 else -1.0

    # fora da influência
    if d_surf >= rho0:
        return 0.0, 0.0, d_surf

    # dentro da influência
    mag = (1.0/d_surf - 1.0/rho0) * (1.0/(d_surf*d_surf))

    mag_rep = k_rep * mag
    Fx_rep = mag_rep * ex
    Fy_rep = mag_rep * ey

    mag_rot = k_rot * mag
    tx = -ey * cross_sign
    ty =  ex * cross_sign
    Fx_rot = mag_rot * tx
    Fy_rot = mag_rot * ty

    return Fx_rep + Fx_rot, Fy_rep + Fy_rot, d_surf

# Parâmetros
@dataclass
class Params:
    # robô
    WHEEL_RADIUS: float = 0.0975
    AXLE_LENGTH:  float = 0.331
    LEFT_SIGN:    int   = 1
    RIGHT_SIGN:   int   = 1

    V_MAX: float = 0.8
    W_MAX: float = 2.4
    WHEEL_OMEGA_MAX: float = 12.0

    GOAL_TOL: float = 0.10
    ROBOT_RADIUS: float = 0.20
    OBSTACLE_RADIUS: float = 0.50

    # retângulo de sub-goals no goal
    GOAL_WIDTH:  float = 18.0
    GOAL_HEIGHT: float = 1.0

    # ganhos do campo
    K_ATT: float = 1.0
    K_REP: float = 10.0
    RHO_0: float = 1.0
    K_ROT: float = 20.0

    # mapeamento força -> (v,w)
    K_V: float = 1.0
    K_W: float = 2.2

    # execução
    TIMEOUT_S: float = 120.0


# Controle
class PotentialFieldController:
    def __init__(self, params: Params):
        self.p = params
        self.r_clear_env = self.p.ROBOT_RADIUS + self.p.OBSTACLE_RADIUS
        self.r_clear_rr  = 2.0 * self.p.ROBOT_RADIUS

    def compute_wheels(self, sim, robot, obstacle_handles, robots):
        """
        Retorna (wl, wr) já saturados e com sinais.
        """
        x, y, yaw = robot.pose
        sx, sy = robot.subgoal

        # 1) Atrativo
        Fx = self.p.K_ATT * (sx - x)
        Fy = self.p.K_ATT * (sy - y)

        # 2) Repulsão + rotacional dos pilares
        for h in obstacle_handles:
            ox, oy, _ = sim.getObjectPosition(h, sim.handle_world)
            fxi, fyi, _ = repulsive_rotational_surface(
                x, y, ox, oy, sx, sy,
                self.r_clear_env,
                self.p.RHO_0,
                self.p.K_REP,
                self.p.K_ROT
            )
            Fx += fxi
            Fy += fyi

        # 3) Outros robôs como obstáculos móveis
        for other in robots:
            if other is robot:
                continue
            if other.reached:
                continue

            oxr, oyr, _ = other.pose
            fxi, fyi, _ = repulsive_rotational_surface(
                x, y, oxr, oyr, sx, sy,
                self.r_clear_rr,
                self.p.RHO_0,
                self.p.K_REP,
                self.p.K_ROT
            )
            Fx += fxi
            Fy += fyi

        # 4) Força (mundo) -> comando (v,w)
        vx_b, vy_b = world_to_body(Fx, Fy, yaw)

        v_cmd = self.p.K_V * math.hypot(vx_b, vy_b)
        w_cmd = self.p.K_W * math.atan2(vy_b, max(1e-6, vx_b))

        # desacelera perto do sub-goal
        dist_goal = math.hypot(sx - x, sy - y)
        if dist_goal < 2.0 * self.p.GOAL_TOL:
            v_cmd *= 0.4

        # saturações
        v_cmd = saturate(v_cmd, -self.p.V_MAX, self.p.V_MAX)
        w_cmd = saturate(w_cmd, -self.p.W_MAX, self.p.W_MAX)

        # 5) (v,w) -> rodas
        wl, wr = vw_to_wheels(v_cmd, w_cmd, self.p.WHEEL_RADIUS, self.p.AXLE_LENGTH)

        wl = saturate(wl, -self.p.WHEEL_OMEGA_MAX, self.p.WHEEL_OMEGA_MAX) * self.p.LEFT_SIGN
        wr = saturate(wr, -self.p.WHEEL_OMEGA_MAX, self.p.WHEEL_OMEGA_MAX) * self.p.RIGHT_SIGN

        return wl, wr

# Robô (objeto)
class PioneerRobot:
    def __init__(self, sim, idx: int, name_suffix: str, params: Params):
        self.sim = sim
        self.p = params
        self.idx = idx  # 1..N (pra log)

        self.rob_h = sim.getObject('/Pioneer_p3dx' + name_suffix)
        self.lm_h  = sim.getObject('/Pioneer_p3dx_leftMotor' + name_suffix)
        self.rm_h  = sim.getObject('/Pioneer_p3dx_rightMotor' + name_suffix)

        self.traj = []
        self.reached = False
        self.pose = (0.0, 0.0, 0.0)   # (x,y,yaw)
        self.subgoal = (0.0, 0.0)     # (sx,sy)

    def log(self, msg):
        print(f"[R{self.idx}] {msg}")

    def set_subgoal(self, sx, sy):
        self.subgoal = (sx, sy)

    def read_pose(self):
        pose = self.sim.getObjectPose(self.rob_h, self.sim.handle_world)
        x, y = pose[0], pose[1]
        yaw = yaw_from_quaternion(pose[3], pose[4], pose[5], pose[6])
        self.pose = (x, y, yaw)
        self.traj.append((x, y))

    def stop(self):
        self.sim.setJointTargetVelocity(self.lm_h, 0.0)
        self.sim.setJointTargetVelocity(self.rm_h, 0.0)

    def apply_wheels(self, wl, wr):
        self.sim.setJointTargetVelocity(self.lm_h, wl)
        self.sim.setJointTargetVelocity(self.rm_h, wr)

    def check_reached(self):
        if self.reached:
            return True

        sx, sy = self.subgoal
        x, y, _ = self.pose

        if math.hypot(sx - x, sy - y) <= self.p.GOAL_TOL:
            self.reached = True
            self.stop()
            self.log("Chegou ao sub-goal.")
            return True

        return False


# Simulação
class SwarmSimulation:
    def __init__(self, N=6, params=None):
        self.N = N
        self.p = params if params is not None else Params()

        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.client.setStepping(True)

        self.controller = PotentialFieldController(self.p)

        self.robots = []
        self.goal_h = None
        self.obstacle_handles = []
        self.obstacles_xy = []
        self.gx = 0.0
        self.gy = 0.0

    def setup(self):
        # robôs
        self.robots = []
        for i in range(self.N):
            suffix = '' if i == 0 else str(i + 1)
            self.robots.append(PioneerRobot(self.sim, idx=i+1, name_suffix=suffix, params=self.p))

        # goal
        self.goal_h = self.sim.getObject('/Goal')

        # start sim
        if self.sim.getSimulationState() == self.sim.simulation_stopped:
            self.sim.startSimulation()

        # obstáculos
        self.obstacle_handles = [
            self.sim.getObject('/80cmHighPillar25cm0'),
            self.sim.getObject('/80cmHighPillar25cm1'),
            self.sim.getObject('/80cmHighPillar25cm2'),
            self.sim.getObject('/80cmHighPillar25cm3'),
        ]

        self.obstacles_xy = []
        for h in self.obstacle_handles:
            ox, oy, _ = self.sim.getObjectPosition(h, self.sim.handle_world)
            self.obstacles_xy.append((ox, oy))

        # subgoals
        self.gx, self.gy, _ = self.sim.getObjectPosition(self.goal_h, self.sim.handle_world)

        spacing_x = self.p.GOAL_WIDTH / (self.N - 1) if self.N > 1 else 0.0
        for k, rob in enumerate(self.robots):
            sx = self.gx + (k * spacing_x) - self.p.GOAL_WIDTH/2
            sy = self.gy
            rob.set_subgoal(sx, sy)

    def all_reached(self):
        return all(r.reached for r in self.robots)

    def stop_all(self):
        for r in self.robots:
            r.stop()

    def run(self):
        self.setup()
        start_time = time.time()

        while True:
            self.client.step()

            # 1) ler poses
            for r in self.robots:
                r.read_pose()

            # 2) checar chegada
            for r in self.robots:
                r.check_reached()

            # 3) se todo mundo chegou, fim
            if self.all_reached():
                print("Todos chegaram. Encerrando.")
                break

            # 4) controle
            for r in self.robots:
                if r.reached:
                    continue
                wl, wr = self.controller.compute_wheels(
                    sim=self.sim,
                    robot=r,
                    obstacle_handles=self.obstacle_handles,
                    robots=self.robots
                )
                r.apply_wheels(wl, wr)

            # 5) timeout
            if time.time() - start_time > self.p.TIMEOUT_S:
                print("Timeout. Encerrando.")
                break

        self.stop_all()
        self.sim.stopSimulation()
        self.plot()

    def plot(self):
        fig, ax = plt.subplots()

        r_clear_env = self.p.ROBOT_RADIUS + self.p.OBSTACLE_RADIUS

        # obstáculos + raio influência
        for (ox, oy) in self.obstacles_xy:
            ax.add_patch(plt.Circle((ox, oy), self.p.OBSTACLE_RADIUS, fill=False, color='k'))
            ax.add_patch(plt.Circle((ox, oy), r_clear_env + self.p.RHO_0,
                                    fill=False, linestyle='--', alpha=0.5))

        # trajetórias
        for i, r in enumerate(self.robots):
            if not r.traj:
                continue
            xs = [p[0] for p in r.traj]
            ys = [p[1] for p in r.traj]
            ax.plot(xs, ys, label=f'robô {i+1}')
            ax.plot(xs[0], ys[0], 'x', markersize=6)
            ax.plot(xs[-1], ys[-1], 'o', markersize=5)

        # goal + retângulo
        ax.plot(self.gx, self.gy, '*', markersize=14, label='goal (centro)')
        ax.add_patch(plt.Rectangle((self.gx - self.p.GOAL_WIDTH/2, self.gy - self.p.GOAL_HEIGHT/2),
                                   self.p.GOAL_WIDTH, self.p.GOAL_HEIGHT,
                                   fill=False, linestyle='-', alpha=0.8))

        # subgoals
        for r in self.robots:
            sx, sy = r.subgoal
            ax.plot(sx, sy, '^', markersize=7)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'Trajetórias de {self.N} Pioneers')
        ax.axis('equal')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

# Execução
def main():
    sim = SwarmSimulation(N=10, params=Params())
    sim.run()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERRO]", e)