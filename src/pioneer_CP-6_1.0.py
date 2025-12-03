import math
import numpy as np
from dataclasses import dataclass
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt
import time

@dataclass
class RobotParams:
    WHEEL_RADIUS: float = 0.0975
    AXLE_LENGTH: float = 0.331
    LEFT_SIGN: int = 1
    RIGHT_SIGN: int = 1
    V_MAX: float = 0.40
    W_MAX: float = 1.2
    WHEEL_OMEGA_MAX: float = 12.0
    GOAL_TOL: float = 0.10
    ROBOT_RADIUS: float = 0.20

@dataclass
class WorldParams:
    OBSTACLE_RADIUS: float = 0.50   # raio físico do pilar
    OBSTACLE_PREFIX: str = '80cmHighPillar25cm'  # para auto-descobrir os obstáculos
    GOAL_WIDTH: float = 10.0       # largura do goal (retângulo)
    GOAL_HEIGHT: float = 1.0      # altura do goal (retângulo)

@dataclass
class FieldGains:
    # Campo potencial
    K_ATT: float = 1.0  # força de atração
    K_REP: float = 3.5  # força de repulsão
    RHO_0: float = 1.0  # janela de influência sobre d_surf
    K_ROT: float = 0.7  # rotacional/tangencial p/ contornar
    # Mapeamento vetor -> (v,w)
    K_V: float = 1.0    # mapeamento campo -> v
    K_W: float = 2.2    # mapeamento campo -> w

def yaw_from_quaternion(qx, qy, qz, qw) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

def saturate(x, a, b):
    return max(min(x, b), a)

def world_vec_to_body(vx_w, vy_w, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c * vx_w - s * vy_w, s * vx_w + c * vy_w

def vw_to_wheel_omegas(v, w, r, L):
    wr = (2.0 * v + w * L) / (2.0 * r)
    wl = (2.0 * v - w * L) / (2.0 * r)
    return wl, wr

def attractive_field(px, py, gx, gy, K_ATT):
    return K_ATT * (gx - px), K_ATT * (gy - py)

def repulsive_rotational_surface(px, py, ox, oy, gx, gy,    # p = posição do robô, g = posição do goal, o = posição do obstáculo, r_clear = raio do robô + raio do obstaculo
                                 K_REP, RHO_0, K_ROT,
                                 r_clear):

    dx, dy = px - ox, py - oy
    d = math.hypot(dx, dy)  # distância euclidiana entre o robô e o obstaculo
    
    d_surf = d - r_clear    # distância até a superficie do obstaculo
    ex, ey = dx / d, dy / d # vetor: obstaculo -> robô unitário

    # vetor obstáculo -> goal
    ogx, ogy = gx - ox, gy - oy

    cross_sign = math.copysign(1.0, ex * ogy - ey * ogx)  # Determina a direção da rotação (1 para anti-horário, -1 para horário)
    """
    cross_sign vai calcular o sinal da rotação (horária ou anti-horária) do robô em torno do obstáculo,
    com base no produto vetorial entre o vetor obstáculo -> robô e o vetor obstáculo -> goal.
    """

    # fora da influência
    if d_surf >= RHO_0:
        return 0.0, 0.0, d_surf

    # dentro da janela de influência
    mag_rep = K_REP * (1.0/d_surf - 1.0/RHO_0) * (1.0/(d_surf*d_surf))
    Fx_rep, Fy_rep = mag_rep * ex, mag_rep * ey

    mag_rot = K_ROT * (1.0/d_surf - 1.0/RHO_0) * (1.0/(d_surf*d_surf))
    tx, ty = -ey * cross_sign, ex * cross_sign
    Fx_rot, Fy_rot = mag_rot * tx, mag_rot * ty

    return Fx_rep + Fx_rot, Fy_rep + Fy_rot, d_surf

def discover_obstacles(sim, prefix) -> list:
    handles = []

    try:
        shape_type = sim.object_shape_type
    except AttributeError:
        shape_type = 4

    try:
        objs = sim.getObjectsInTree(sim.handle_scene, shape_type, 0)
    except Exception:
        try:
            objs = sim.getObjects(shape_type, sim.handle_all)
        except TypeError:
            objs = sim.getObjects(shape_type)

    if isinstance(objs, int):
        objs = [objs]
    if not isinstance(objs, (list, tuple)):
        objs = list(objs)

    for h in objs:
        try:
            alias = sim.getObjectAlias(h, 0)
        except Exception:
            alias = sim.getObjectAlias(h)
        if isinstance(alias, bytes):
            alias = alias.decode('utf-8', errors='ignore')

        if alias.startswith(prefix) or alias.endswith('/' + prefix) or f'/{prefix}' in alias:
            handles.append(h)

    return handles

def compute_control_for_robot(rx, ry, yaw,
                              gx, gy,
                              obstacle_handles,
                              sim,
                              gains: FieldGains,
                              rp: RobotParams,
                              r_clear_env: float,
                              other_robots_pos=None,
                              r_clear_rr: float = None):
    """
    Calcula (v_cmd, w_cmd) para um robô dado sua pose,
    obstáculos estáticos e outros robôs como obstáculos móveis.
    """
    # Campo atrativo
    Fx_att, Fy_att = attractive_field(rx, ry, gx, gy, gains.K_ATT)

    Fx_rep_sum, Fy_rep_sum = 0.0, 0.0
    d_surf_min = float('inf')

    # Obstáculos estáticos (pilares)
    for h in obstacle_handles:
        ox, oy, _ = sim.getObjectPosition(h, sim.handle_world)
        Fx_i, Fy_i, d_surf_i = repulsive_rotational_surface(
            rx, ry, ox, oy, gx, gy,
            gains.K_REP, gains.RHO_0, gains.K_ROT,
            r_clear_env
        )
        Fx_rep_sum += Fx_i
        Fy_rep_sum += Fy_i
        d_surf_min = min(d_surf_min, d_surf_i)

    # Outros robôs como obstáculos móveis (lista)
    if other_robots_pos is not None and r_clear_rr is not None:
        for (ox_r, oy_r) in other_robots_pos:
            Fx_r, Fy_r, d_surf_r = repulsive_rotational_surface(
                rx, ry, ox_r, oy_r, gx, gy,
                gains.K_REP, gains.RHO_0, gains.K_ROT,
                r_clear_rr
            )
            Fx_rep_sum += Fx_r
            Fy_rep_sum += Fy_r
            d_surf_min = min(d_surf_min, d_surf_r)

    Fx, Fy = Fx_att + Fx_rep_sum, Fy_att + Fy_rep_sum

    # vetor desejado (mundo) -> (v,w) no corpo
    vx_b, vy_b = world_vec_to_body(Fx, Fy, yaw)
    v_cmd = gains.K_V * math.hypot(vx_b, vy_b)
    w_cmd = gains.K_W * math.atan2(vy_b, max(1e-6, vx_b))

    return v_cmd, w_cmd, d_surf_min

def main():
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    client.setStepping(True)

    rp = RobotParams()
    wp = WorldParams()
    gains = FieldGains()

    N_ROBOTS = 6    # Quantidade de robôs

    robots = []

    for i in range(N_ROBOTS):
        suffix = '' if i == 0 else str(i + 1)
        robot_name = '/Pioneer_p3dx' + suffix
        left_name  = '/Pioneer_p3dx_leftMotor' + suffix
        right_name = '/Pioneer_p3dx_rightMotor' + suffix

        robot_handle = sim.getObject(robot_name)
        left_motor   = sim.getObject(left_name)
        right_motor  = sim.getObject(right_name)

        robots.append({
            'handle': robot_handle,
            'left_motor': left_motor,
            'right_motor': right_motor,
            'traj': [],
            'reached': False,
            'pose': (0.0, 0.0, 0.0),
            'subgoal': (0.0, 0.0)
        })

    goal = sim.getObject('/Goal')

    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()

    # Obstáculos estáticos
    obstacle_handles = [
        sim.getObject('/80cmHighPillar25cm0'),
        sim.getObject('/80cmHighPillar25cm1'),
        sim.getObject('/80cmHighPillar25cm2'),
        sim.getObject('/80cmHighPillar25cm3'),
    ]

    # Guarda as posições dos obstáculos
    obstacles_xy = []
    for h in obstacle_handles:
        ox, oy, _ = sim.getObjectPosition(h, sim.handle_world)
        obstacles_xy.append((ox, oy))

    # Distância de segurança entre robô e obstáculo
    r_clear_env = rp.ROBOT_RADIUS + wp.OBSTACLE_RADIUS

    # Distância de segurança entre robôs
    r_clear_rr = 2.0 * rp.ROBOT_RADIUS

    # posição do goal
    gx, gy, _ = sim.getObjectPosition(goal, sim.handle_world)

    # Definindo a posição dos sub-goals ao longo do retângulo
    goal_width = wp.GOAL_WIDTH
    goal_height = wp.GOAL_HEIGHT
    spacing_x = goal_width / (N_ROBOTS - 1)
    spacing_y = 0

    for k in range(N_ROBOTS):
        sx = gx + (k * spacing_x) - goal_width / 2  # Ajusta para ficar dentro do retângulo
        sy = gy  # Todos os sub-goals na mesma linha, sem alteração no eixo y
        robots[k]['subgoal'] = (sx, sy)

    start_time = time.time()

    while True:
        client.step()

        # Ler pose de todos e registrar trajetória
        for rob in robots:
            pose = sim.getObjectPose(rob['handle'], sim.handle_world)
            x, y = pose[0], pose[1]
            yaw = yaw_from_quaternion(pose[3], pose[4], pose[5], pose[6])
            rob['pose'] = (x, y, yaw)
            rob['traj'].append((x, y))

        # Verificar chegada e parar robôs que chegaram
        for rob in robots:
            if rob['reached']:
                continue
            sx, sy = rob['subgoal']
            x, y, _ = rob['pose']
            dist_goal = math.hypot(sx - x, sy - y)

            if dist_goal <= rp.GOAL_TOL:
                rob['reached'] = True
                sim.setJointTargetVelocity(rob['left_motor'], 0.0)
                sim.setJointTargetVelocity(rob['right_motor'], 0.0)
                print('Um robô chegou ao seu sub-goal em', sx, sy)

        # Se todos chegaram, encerra
        if all(rob['reached'] for rob in robots):
            print('Todos os robôs chegaram aos seus sub-goals. Parando simulação.')
            break

        # Controle de cada robô
        for i, rob in enumerate(robots):
            if rob['reached']:
                continue

            x, y, yaw = rob['pose']
            sx, sy = rob['subgoal']

            # lista de posições dos outros robôs "ativos"
            other_positions = []
            for j, other in enumerate(robots):
                if j == i:
                    continue
                if other['reached']:
                    continue
                ox, oy, _ = other['pose']
                other_positions.append((ox, oy))

            v_cmd, w_cmd, d_surf_min = compute_control_for_robot(
                x, y, yaw,
                sx, sy,
                obstacle_handles,
                sim,
                gains,
                rp,
                r_clear_env,
                other_robots_pos=other_positions if other_positions else None,
                r_clear_rr=r_clear_rr
            )

            dist_goal = math.hypot(sx - x, sy - y)
            if dist_goal < 2.0 * rp.GOAL_TOL:
                v_cmd *= 0.6

            v_cmd = saturate(v_cmd, -rp.V_MAX, rp.V_MAX)
            w_cmd = saturate(w_cmd, -rp.W_MAX, rp.W_MAX)

            wl, wr = vw_to_wheel_omegas(v_cmd, w_cmd, rp.WHEEL_RADIUS, rp.AXLE_LENGTH)
            wl = saturate(wl, -rp.WHEEL_OMEGA_MAX, rp.WHEEL_OMEGA_MAX) * rp.LEFT_SIGN
            wr = saturate(wr, -rp.WHEEL_OMEGA_MAX, rp.WHEEL_OMEGA_MAX) * rp.RIGHT_SIGN

            sim.setJointTargetVelocity(rob['left_motor'], wl)
            sim.setJointTargetVelocity(rob['right_motor'], wr)

        # Tempo limite de simulação
        elapsed_time = time.time() - start_time
        if elapsed_time > 120:
            print('Tempo limite alcançado. Finalizando simulação.')
            break
    # Encerrar e plotar trajetórias
    for rob in robots:
        sim.setJointTargetVelocity(rob['left_motor'], 0.0)
        sim.setJointTargetVelocity(rob['right_motor'], 0.0)

    sim.stopSimulation()

    # Plotando as trajetórias
    fig, ax = plt.subplots()

    # Obstáculos físicos
    for (ox, oy) in obstacles_xy:
        circ_obs = plt.Circle((ox, oy),
                              wp.OBSTACLE_RADIUS,
                              fill=False,
                              color='k')
        ax.add_patch(circ_obs)

        # Raio de influência do campo repulsivo
        raio_influencia = r_clear_env + gains.RHO_0
        circ_rep = plt.Circle((ox, oy),
                              raio_influencia,
                              fill=False,
                              linestyle='--',
                              alpha=0.5)
        ax.add_patch(circ_rep)

    # Trajetórias
    for idx, rob in enumerate(robots):
        if not rob['traj']:
            continue
        xs = [p[0] for p in rob['traj']]
        ys = [p[1] for p in rob['traj']]
        ax.plot(xs, ys, label=f'trajetória robô {idx+1}')
        ax.plot(xs[0], ys[0], 'x', markersize=6)
        ax.plot(xs[-1], ys[-1], 'o', markersize=5)

    # Goal global
    ax.plot(gx, gy, '*', markersize=14, label='goal (centro)')
    rect_goal = plt.Rectangle((gx - wp.GOAL_WIDTH / 2, gy - wp.GOAL_HEIGHT / 2),
                              wp.GOAL_WIDTH, wp.GOAL_HEIGHT, fill=False, linestyle='-', alpha=0.8)
    ax.add_patch(rect_goal)

    # Sub-goals
    for k, rob in enumerate(robots):
        sx, sy = rob['subgoal']
        ax.plot(sx, sy, '^', markersize=7, label=f'sub-goal robô {k+1}')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Trajetórias de {N_ROBOTS} Pioneers com repulsão mútua e sub-goals em retângulo')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('[ERRO]', e)