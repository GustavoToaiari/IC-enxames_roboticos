import math
from dataclasses import dataclass
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Parâmetros
@dataclass
class Robo_Parametros:
    RAIO_rodas: float = 0.0975    # [m]
    DIST_rodas: float = 0.381  # [m]
    V_max: float = 0.40            # [m/s]
    W_max: float = 1.2             # [rad/s]
    W_roda_max: float = 12.0  # [rad/s]
    GOAL_tolerancia: float = 0.10         # [m]
    RAIO_robo: float = 0.20     # [m]

@dataclass
class Global_Parametros:
    RAIO_obstaculo: float = 1.0   # [m]

@dataclass
class Campo_Ganhos:
    # Campo potencial
    K_atracao: float = 1.0
    K_repulsao: float = 0.3
    RHO_0: float = 0.5  # [m] janela de influência SOBRE d_surf
    K_rotacao: float = 1.0  # ganho tangencial para contorno
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
    def __init__(self, gains: Campo_Ganhos, r_clear: float):
        self.g = gains
        self.r_clear = r_clear

    def attractive(self, px, py, gx, gy):
        Fx = self.g.K_atracao * (gx - px)
        Fy = self.g.K_atracao * (gy - py)
        return Fx, Fy

    def repulsive_rotational_surface(self, px, py, ox, oy, gx, gy):
        """
        Usando distância à SUPERFÍCIE:
          d_surf = ||p - o|| - r_clear
        """
        dx, dy = px - ox, py - oy
        d = math.hypot(dx, dy)
        if d <= 1e-6:
            return 0.0, 0.0, -float('inf')  # patológico

        d_surf = d - self.r_clear
        ex, ey = dx / d, dy / d  # obs->robô unitário

        # Sentido de giro definido por obs->goal
        ogx, ogy = gx - ox, gy - oy
        cross_sign = math.copysign(1.0, ex * ogy - ey * ogx)

        # Encostou/entrou
        if d_surf <= 0.0:
            mag_rep = self.g.K_repulsao * (1.0 / max(1e-3, -d_surf))
            Fx_rep, Fy_rep = mag_rep * ex, mag_rep * ey
            tx, ty = -ey * cross_sign, ex * cross_sign
            Fx_rot, Fy_rot = self.g.K_rotacao * tx, self.g.K_rotacao * ty
            return Fx_rep + Fx_rot, Fy_rep + Fy_rot, d_surf

        # Fora da janela
        if d_surf >= self.g.RHO_0:
            return 0.0, 0.0, d_surf

        # Dentro da janela: Khatib sobre d_surf
        mag = (1.0/d_surf - 1.0/self.g.RHO_0) * (1.0/(d_surf*d_surf))
        Fx_rep, Fy_rep = self.g.K_repulsao * mag * ex, self.g.K_repulsao * mag * ey
        tx, ty = -ey * cross_sign, ex * cross_sign
        Fx_rot, Fy_rot = self.g.K_rotacao * mag * tx, self.g.K_rotacao * mag * ty

        return Fx_rep + Fx_rot, Fy_rep + Fy_rot, d_surf


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
# Navegador (orquestra o laço)
# ======================

class Navigator:
    def __init__(self, sim: SimInterface, robot: DifferentialDriveRobot,
                 goal_path='/Goal', obstacle_path='/80cmHighPillar25cm',
                 world: Global_Parametros = Global_Parametros(),
                 gains: Campo_Ganhos = Campo_Ganhos()):
        self.sim = sim
        self.robot = robot
        self.world = world
        self.gains = gains

        self.goal = self.sim.get_handle(goal_path)
        self.obstacle = self.sim.get_handle(obstacle_path)

        r_clear = robot.p.RAIO_robo + world.RAIO_obstaculo
        print(f'[INFO] Clearance geométrico: r_clear = {r_clear:.3f} m')

        self.fields = PotentialFieldPlanner(self.gains, r_clear)
        self.mapper = VelocityMapper(self.gains, robot.p)

    def read_goal(self):
        gx, gy, _ = self.sim.get_position(self.goal)
        return gx, gy

    def read_obstacle(self):
        ox, oy, _ = self.sim.get_position(self.obstacle)
        return ox, oy

    def reached_goal(self, x, y, gx, gy):
        return math.hypot(gx - x, gy - y) <= self.robot.p.GOAL_tolerancia

    def run(self):
        # inicia simulação se necessário
        if self.sim.sim.getSimulationState() == self.sim.simulation_stopped:
            self.sim.sim.startSimulation()

        while True:
            self.sim.step()

            # estados
            rx, ry, yaw = self.robot.read_state()
            gx, gy = self.read_goal()
            ox, oy = self.read_obstacle()

            # chegou?
            if self.reached_goal(rx, ry, gx, gy):
                self.robot.stop()
                print('[INFO] Meta atingida. Parando.')
                break

            # campos
            Fx_att, Fy_att = self.fields.attractive(rx, ry, gx, gy)
            Fx_rep, Fy_rep, d_surf = self.fields.repulsive_rotational_surface(
                rx, ry, ox, oy, gx, gy
            )
            Fx, Fy = Fx_att + Fx_rep, Fy_att + Fy_rep

            # vetor -> (v,w)
            dist_goal = math.hypot(gx - rx, gy - ry)
            near_goal = dist_goal < 2.0 * self.robot.p.GOAL_tolerancia
            emergency = d_surf <= 0.0

            v_cmd, w_cmd = self.mapper.vec_to_vw(Fx, Fy, yaw,
                                                 near_goal=near_goal,
                                                 emergency=emergency)

            # (v,w) -> rodas
            wl, wr = self.mapper.vw_to_wheels(v_cmd, w_cmd)
            self.robot.command_wheels(wl, wr)

        self.sim.stop()


# ======================
# main
# ======================

def main():
    rp = Robo_Parametros()
    wp = Global_Parametros()
    gains = Campo_Ganhos()

    sim = SimInterface()
    robot = DifferentialDriveRobot(sim, rp)
    nav = Navigator(sim, robot, world=wp, gains=gains)
    nav.run()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('[ERRO]', e)
