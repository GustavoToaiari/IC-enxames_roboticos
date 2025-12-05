import math
from dataclasses import dataclass
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt

# Parâmetros
@dataclass
class Robo_Parametros:
    RAIO_rodas: float = 0.0975    # [m]
    DIST_rodas: float = 0.381     # [m]
    V_max: float = 0.80           # [m/s]
    W_max: float = 2.4            # [rad/s]
    W_roda_max: float = 12.0      # [rad/s]
    GOAL_tolerancia: float = 0.10 # [m]
    RAIO_robo: float = 0.20       # [m]

@dataclass
class Global_Parametros:
    RAIO_obstaculo: float = 0.5   # [m]

@dataclass
class Campo_Ganhos:
    # Campo potencial - goal/obstáculo
    K_atracao: float = 1.0
    K_repulsao: float = 10
    RHO_0: float = 1.0
    K_rotacao: float = 20.0

    # Repulsão entre robôs
    K_repulsao_robos: float = 1.0
    RHO_0_robos: float = 0.8
    K_rotacao_robos: float = 0.0  # normalmente 0

    # Mapeamento vetor -> (v,w)
    K_V: float = 1.0
    K_W: float = 2.2

    # Emergência
    K_W_EMERG: float = 2.5
    BACKUP_V: float = -0.05


def saturate(x, a, b):
    return max(min(x, b), a)

def yaw_from_quaternion(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def world_to_body(vx_w, vy_w, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c*vx_w - s*vy_w, s*vx_w + c*vy_w

# ======================
# Interface com CoppeliaSim
# ======================

class SimInterface:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.client.setStepping(True)

    def get_handle(self, path: str):
        return self.sim.getObject(path)

    def start_if_needed(self):
        if self.sim.getSimulationState() == self.simulation_stopped:
            self.sim.startSimulation()

    @property
    def simulation_stopped(self):
        return self.sim.simulation_stopped

    def step(self):
        self.client.step()

    def stop(self):
        self.sim.stopSimulation()

    # Wrappers de leitura
    def get_pose(self, handle):
        return self.sim.getObjectPose(handle, self.sim.handle_world)

    def get_position(self, handle):
        return self.sim.getObjectPosition(handle, self.sim.handle_world)

    # Wrappers de escrita
    def set_joint_velocity(self, joint, vel):
        self.sim.setJointTargetVelocity(joint, vel)


# ======================
# Robô diferencial (Pioneer)
# ======================

class DifferentialDriveRobot:
    def __init__(self, sim: SimInterface, params: Robo_Parametros,
                 robot_path='/Pioneer_p3dx',
                 left_motor_path='/Pioneer_p3dx_leftMotor',
                 right_motor_path='/Pioneer_p3dx_rightMotor'):
        self.sim = sim
        self.p = params
        self.handle = self.sim.get_handle(robot_path)
        self.left = self.sim.get_handle(left_motor_path)
        self.right = self.sim.get_handle(right_motor_path)

    def read_state(self):
        pose = self.sim.get_pose(self.handle)  # [x,y,z,qx,qy,qz,qw]
        x, y = pose[0], pose[1]
        yaw = yaw_from_quaternion(pose[3], pose[4], pose[5], pose[6])
        return x, y, yaw

    def command_wheels(self, wl, wr):
        # Satura e aplica sinais de lado
        wl = saturate(wl, -self.p.W_roda_max, self.p.W_roda_max)
        wr = saturate(wr, -self.p.W_roda_max, self.p.W_roda_max)
        self.sim.set_joint_velocity(self.left, wl)
        self.sim.set_joint_velocity(self.right, wr)

    def stop(self):
        self.sim.set_joint_velocity(self.left, 0.0)
        self.sim.set_joint_velocity(self.right, 0.0)


# ======================
# Campos Potenciais (atrativo + repulsivo/rotacional)
# ======================

class PotentialFieldPlanner:
    def __init__(self, gains: Campo_Ganhos,
                 r_clear_obst: float,
                 r_clear_robos: float):
        self.g = gains
        self.r_clear_obst = r_clear_obst
        self.r_clear_robos = r_clear_robos

    def attractive(self, px, py, gx, gy):
        Fx = self.g.K_atracao * (gx - px)
        Fy = self.g.K_atracao * (gy - py)
        return Fx, Fy

    def repulsive_rotational_surface(self, px, py, ox, oy, gx, gy):
        dx, dy = px - ox, py - oy
        d = math.hypot(dx, dy)
        if d <= 1e-6:
            return 0.0, 0.0, -float('inf')

        d_surf = d - self.r_clear_obst
        ex, ey = dx / d, dy / d

        ogx, ogy = gx - ox, gy - oy
        cross_sign = math.copysign(1.0, ex * ogy - ey * ogx)

        if d_surf <= 0.0:
            mag_rep = self.g.K_repulsao * (1.0 / max(1e-3, -d_surf))
            Fx_rep, Fy_rep = mag_rep * ex, mag_rep * ey
            tx, ty = -ey * cross_sign, ex * cross_sign
            Fx_rot, Fy_rot = self.g.K_rotacao * tx, self.g.K_rotacao * ty
            return Fx_rep + Fx_rot, Fy_rep + Fy_rot, d_surf

        if d_surf >= self.g.RHO_0:
            return 0.0, 0.0, d_surf

        mag = (1.0/d_surf - 1.0/self.g.RHO_0) * (1.0/(d_surf*d_surf))
        Fx_rep, Fy_rep = self.g.K_repulsao * mag * ex, self.g.K_repulsao * mag * ey
        tx, ty = -ey * cross_sign, ex * cross_sign
        Fx_rot, Fy_rot = self.g.K_rotacao * mag * tx, self.g.K_rotacao * mag * ty

        return Fx_rep + Fx_rot, Fy_rep + Fy_rot, d_surf

    def repulsive_robot_surface(self, px, py, rx, ry):
        dx, dy = px - rx, py - ry
        d = math.hypot(dx, dy)
        if d <= 1e-6:
            return 0.0, 0.0, -float('inf')

        d_surf = d - self.r_clear_robos
        ex, ey = dx / d, dy / d

        if d_surf <= 0.0:
            mag_rep = self.g.K_repulsao_robos * (1.0 / max(1e-3, -d_surf))
            Fx_rep, Fy_rep = mag_rep * ex, mag_rep * ey
            return Fx_rep, Fy_rep, d_surf

        if d_surf >= self.g.RHO_0_robos:
            return 0.0, 0.0, d_surf

        mag = (1.0/d_surf - 1.0/self.g.RHO_0_robos) * (1.0/(d_surf*d_surf))
        Fx_rep = self.g.K_repulsao_robos * mag * ex
        Fy_rep = self.g.K_repulsao_robos * mag * ey

        return Fx_rep, Fy_rep, d_surf


# ======================
# Mapeamento de vetor (mundo) -> (v,w) -> (wl, wr)
# ======================

class VelocityMapper:
    def __init__(self, gains: Campo_Ganhos, params: Robo_Parametros):
        self.g = gains
        self.p = params

    def vec_to_vw(self, Fx, Fy, yaw, near_goal=False, emergency=False):
        vx_b, vy_b = world_to_body(Fx, Fy, yaw)
        v_cmd = self.g.K_V * math.hypot(vx_b, vy_b)
        w_cmd = self.g.K_W * math.atan2(vy_b, max(1e-6, vx_b))

        if emergency:
            v_cmd = min(0.0, self.g.BACKUP_V)
            w_cmd = saturate(w_cmd * (self.g.K_W_EMERG / max(1e-6, self.g.K_W)),
                             -self.p.W_max, self.p.W_max)

        if near_goal:
            v_cmd *= 0.3

        v_cmd = saturate(v_cmd, -self.p.V_max, self.p.V_max)
        w_cmd = saturate(w_cmd, -self.p.W_max, self.p.W_max)
        return v_cmd, w_cmd

    def vw_to_wheels(self, v, w):
        r, L = self.p.RAIO_rodas, self.p.DIST_rodas
        wr = (2.0*v + w*L) / (2.0*r)
        wl = (2.0*v - w*L) / (2.0*r)
        return wl, wr


# ======================
# Navegador MULTI-ROBÔS
# ======================

class MultiRobotNavigator:
    def __init__(self, sim: SimInterface,
                 robots: list,
                 goal_paths: list,
                 obstacle_paths: list,
                 world: Global_Parametros = Global_Parametros(),
                 gains: Campo_Ganhos = Campo_Ganhos()):

        assert len(robots) == len(goal_paths)

        self.sim = sim
        self.robots = robots
        self.world = world
        self.gains = gains

        self.goals = [self.sim.get_handle(path) for path in goal_paths]
        self.obstacles = [self.sim.get_handle(path) for path in obstacle_paths]

        # Clearance geométrico
        r_clear_obst = robots[0].p.RAIO_robo + world.RAIO_obstaculo
        r_clear_robos = 2.0 * robots[0].p.RAIO_robo

        print(f'[INFO] Clearance obstáculo: {r_clear_obst:.3f} m')
        print(f'[INFO] Clearance robôs: {r_clear_robos:.3f} m')

        # guardar para usar no plot
        self.r_clear_obst = r_clear_obst

        self.fields = PotentialFieldPlanner(self.gains, r_clear_obst, r_clear_robos)
        self.mapper = VelocityMapper(self.gains, robots[0].p)

        # listas de trajetórias (uma por robô)
        self.traj_x = [[] for _ in robots]
        self.traj_y = [[] for _ in robots]


    def read_goal(self, i):
        gx, gy, _ = self.sim.get_position(self.goals[i])
        return gx, gy

    def read_obstacle(self, k):
        ox, oy, _ = self.sim.get_position(self.obstacles[k])
        return ox, oy

    def reached_goal(self, robot: DifferentialDriveRobot, x, y, gx, gy):
        return math.hypot(gx - x, gy - y) <= robot.p.GOAL_tolerancia
    
    def plot_results(self):
        fig, ax = plt.subplots()

        # Trajetórias, posição inicial e final
        for i in range(len(self.robots)):
            xs = self.traj_x[i]
            ys = self.traj_y[i]
            if not xs:
                continue

            ax.plot(xs, ys, label=f'Robô {i}')
            # posição inicial
            ax.scatter(xs[0], ys[0], marker='o', s=40)
            ax.text(xs[0], ys[0], f'R{i}_ini', fontsize=8, va='bottom')
            # posição final
            ax.scatter(xs[-1], ys[-1], marker='x', s=50)
            ax.text(xs[-1], ys[-1], f'R{i}_fin', fontsize=8, va='top')

        # Goals
        for i, goal_h in enumerate(self.goals):
            gx, gy, _ = self.sim.get_position(goal_h)
            ax.scatter(gx, gy, marker='*', s=80)
            ax.text(gx, gy, f'G{i}', fontsize=9, va='bottom', ha='center')

        # Obstáculos e raio do campo
        for k, obst_h in enumerate(self.obstacles):
            ox, oy, _ = self.sim.get_position(obst_h)

            # círculo do obstáculo (raio físico)
            obst_circle = plt.Circle(
                (ox, oy),
                self.world.RAIO_obstaculo,
                fill=False,
                linewidth=1.5
            )
            ax.add_patch(obst_circle)
            ax.text(ox, oy, f'O{k}', fontsize=8, ha='center', va='center')

            # círculo do campo de influência: r_clear_obst + RHO_0
            field_radius = self.r_clear_obst + self.gains.RHO_0
            field_circle = plt.Circle(
                (ox, oy),
                field_radius,
                fill=False,
                linestyle='--',
                linewidth=1.0
            )
            ax.add_patch(field_circle)

        ax.set_aspect('equal', 'box')
        ax.grid(True)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Trajetórias dos robôs, obstáculos e campos de influência')
        ax.legend()
        plt.show()

    def run(self):
        # Inicia simulação se necessário
        if self.sim.sim.getSimulationState() == self.sim.simulation_stopped:
            self.sim.sim.startSimulation()

        n = len(self.robots)
        reached = [False] * n

        while True:
            self.sim.step()

            # Se todos chegaram, sai do laço
            if all(reached):
                for r in self.robots:
                    r.stop()
                print('[INFO] Todos os robôs atingiram seus goals. Encerrando laço.')
                break

            # Lê estado de TODOS os robôs uma vez
            estados = [r.read_state() for r in self.robots]

            # Loop em cada robô
            for i, robot in enumerate(self.robots):
                if reached[i]:
                    robot.stop()
                    continue

                rx, ry, yaw = estados[i]

                # salva trajetória
                self.traj_x[i].append(rx)
                self.traj_y[i].append(ry)

                gx, gy = self.read_goal(i)

                # Chegou?
                if self.reached_goal(robot, rx, ry, gx, gy):
                    robot.stop()
                    reached[i] = True
                    print(f'[INFO] Robô {i} atingiu sua meta.')
                    continue

                # Campo atrativo
                Fx_att, Fy_att = self.fields.attractive(rx, ry, gx, gy)

                # Repulsão de obstáculos
                Fx_rep_obs, Fy_rep_obs = 0.0, 0.0
                min_d_surf = float('inf')

                for k in range(len(self.obstacles)):
                    ox, oy = self.read_obstacle(k)
                    Fx_o, Fy_o, d_surf_o = self.fields.repulsive_rotational_surface(
                        rx, ry, ox, oy, gx, gy
                    )
                    Fx_rep_obs += Fx_o
                    Fy_rep_obs += Fy_o
                    if d_surf_o < min_d_surf:
                        min_d_surf = d_surf_o

                # Repulsão entre robôs
                Fx_rep_rob, Fy_rep_rob = 0.0, 0.0
                for j, (rx_j, ry_j, yaw_j) in enumerate(estados):
                    if j == i:
                        continue
                    Fx_r, Fy_r, d_surf_r = self.fields.repulsive_robot_surface(
                        rx, ry, rx_j, ry_j
                    )
                    Fx_rep_rob += Fx_r
                    Fy_rep_rob += Fy_r
                    if d_surf_r < min_d_surf:
                        min_d_surf = d_surf_r

                # Soma total dos campos
                Fx = Fx_att + Fx_rep_obs + Fx_rep_rob
                Fy = Fy_att + Fy_rep_obs + Fy_rep_rob

                # vetor -> (v,w)
                dist_goal = math.hypot(gx - rx, gy - ry)
                near_goal = dist_goal < 2.0 * robot.p.GOAL_tolerancia
                emergency = (min_d_surf <= 0.0)

                v_cmd, w_cmd = self.mapper.vec_to_vw(Fx, Fy, yaw,
                                                     near_goal=near_goal,
                                                     emergency=emergency)

                # (v,w) -> rodas
                wl, wr = self.mapper.vw_to_wheels(v_cmd, w_cmd)
                robot.command_wheels(wl, wr)

        # Depois que sai do laço principal, gera o plot
        self.plot_results()

        # Para a simulação
        self.sim.stop()


# ======================
# main
# ======================

def main():
    rp = Robo_Parametros()
    wp = Global_Parametros()
    gains = Campo_Ganhos()

    sim = SimInterface()

    # ------- AJUSTE AQUI OS NOMES DOS OBJETOS NA CENA -------

    # 6 robôs (exemplo de nomes – adapte aos da sua cena)
    robot_paths = [
        '/Pioneer_p3dx',
        '/Pioneer_p3dx2',
        '/Pioneer_p3dx3',
        '/Pioneer_p3dx4',
        '/Pioneer_p3dx5',
        '/Pioneer_p3dx6',
    ]
    left_motor_paths = [
        '/Pioneer_p3dx_leftMotor',
        '/Pioneer_p3dx_leftMotor2',
        '/Pioneer_p3dx_leftMotor3',
        '/Pioneer_p3dx_leftMotor4',
        '/Pioneer_p3dx_leftMotor5',
        '/Pioneer_p3dx_leftMotor6',
    ]
    right_motor_paths = [
        '/Pioneer_p3dx_rightMotor',
        '/Pioneer_p3dx_rightMotor2',
        '/Pioneer_p3dx_rightMotor3',
        '/Pioneer_p3dx_rightMotor4',
        '/Pioneer_p3dx_rightMotor5',
        '/Pioneer_p3dx_rightMotor6',
    ]

    # 6 goals (um para cada robô)
    goal_paths = [
        '/Goal1',
        '/Goal2',
        '/Goal3',
        '/Goal4',
        '/Goal5',
        '/Goal6',
    ]

    # 4 obstáculos (compartilhados)
    obstacle_paths = [
        '/80cmHighPillar25cm0',
        '/80cmHighPillar25cm1',
        '/80cmHighPillar25cm2',
        '/80cmHighPillar25cm3',
    ]
    # Cria os 6 robôs
    robots = []
    for r_path, l_path, rr_path in zip(robot_paths, left_motor_paths, right_motor_paths):
        robots.append(DifferentialDriveRobot(sim, rp,
                                             robot_path=r_path,
                                             left_motor_path=l_path,
                                             right_motor_path=rr_path))

    # Navegador multi-robôs
    nav = MultiRobotNavigator(sim, robots,
                              goal_paths=goal_paths,
                              obstacle_paths=obstacle_paths,
                              world=wp,
                              gains=gains)
    nav.run()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('[ERRO]', e)
