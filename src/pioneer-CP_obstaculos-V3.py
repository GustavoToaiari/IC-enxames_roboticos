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
    OBSTACLE_PREFIX: str = '80cmHighPillar25cm'  # para auto-descobrir os obstaculos
    GOAL_RADIUS: float = 0.7       # raio do goal GLOBAL (visual / espaço p/ robôs)

@dataclass
class FieldGains:
    # Campo potencial
    K_ATT: float = 1.0  # força de atração
    K_REP: float = 3.5  # força de atração
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
    return c*vx_w - s*vy_w, s*vx_w + c*vy_w

def vw_to_wheel_omegas(v, w, r, L):
    wr = (2.0*v + w*L) / (2.0*r)
    wl = (2.0*v - w*L) / (2.0*r)
    return wl, wr

def attractive_field(px, py, gx, gy, K_ATT):
    return K_ATT * (gx - px), K_ATT * (gy - py)

def repulsive_rotational_surface(px, py, ox, oy, gx, gy,    # p = posição do robô, g = posição do goal, o = posição do obstáculo, r_clear = raio do robô + raio do obstaculo
                                 K_REP, RHO_0, K_ROT,
                                 r_clear):

    dx, dy = px - ox, py - oy
    d = math.hypot(dx, dy)  # distância euclidiana entre o robô e o obstaculo
    
    d_surf = d - r_clear    # distância até a superficie
    ex, ey = dx / d, dy / d # vetor: obstaculo -> robô unitário

    # Calcula o vetor que aponta da posição do obstáculo -> objetivo
    ogx, ogy = gx - ox, gy - oy  # vetor: obstáculo -> goal

    cross_sign = math.copysign(1.0, ex * ogy - ey * ogx)  # Determina a direção da rotação (1 para anti-horário, -1 para horário)
    """
    cross_sign vai calcular o sinal da rotação (horária ou anti-horária) do robô em torno do obstáculo,
    com base no produto vetorial entre o vetor obstáculo -> robô e o vetor obstáculo -> goal.
    """

    # fora da influência
    if d_surf >= RHO_0:
        return 0.0, 0.0, d_surf

    # dentro da janela
    mag_rep = K_REP * (1.0/d_surf - 1.0/RHO_0) * (1.0/(d_surf*d_surf))
    Fx_rep, Fy_rep = mag_rep * ex, mag_rep * ey

    mag_rot = K_ROT * (1.0/d_surf - 1.0/RHO_0) * (1.0/(d_surf*d_surf))
    tx, ty = -ey * cross_sign, ex * cross_sign
    Fx_rot, Fy_rot = mag_rot * tx, mag_rot * ty

    return Fx_rep + Fx_rot, Fy_rep + Fy_rot, d_surf

def discover_obstacles(sim, prefix) -> list:
    handles = []

    # Tenta o caminho recomendado: toda a cena, apenas shapes
    try:
        shape_type = sim.object_shape_type  # constante oficial
    except AttributeError:
        shape_type = 4  # fallback típico para 'shape'

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
                              other_robot_pos=None,
                              r_clear_rr: float = None):
    """
    Calcula (v_cmd, w_cmd) para um robô dado sua pose,
    obstáculos ESTÁTICOS e, opcionalmente, outro robô como obstáculo móvel.
    """
    # Campo atrativo para o (sub)goal
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

    # Outro robô como obstáculo móvel (se fornecido)
    if other_robot_pos is not None and r_clear_rr is not None:
        ox_r, oy_r = other_robot_pos
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

    # Robôs e motores
    robot1 = sim.getObject('/Pioneer_p3dx')
    leftMotor1 = sim.getObject('/Pioneer_p3dx_leftMotor')
    rightMotor1 = sim.getObject('/Pioneer_p3dx_rightMotor')

    robot2 = sim.getObject('/Pioneer_p3dx2')
    leftMotor2 = sim.getObject('/Pioneer_p3dx_leftMotor2')
    rightMotor2 = sim.getObject('/Pioneer_p3dx_rightMotor2')

    goal = sim.getObject('/Goal')

    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()

    rp = RobotParams()
    wp = WorldParams()
    gains = FieldGains()

    # ---- Obstáculos estáticos ----
    obstacle_handles = [
            sim.getObject('/80cmHighPillar25cm0'),
            sim.getObject('/80cmHighPillar25cm1'),
            sim.getObject('/80cmHighPillar25cm2'),
            sim.getObject('/80cmHighPillar25cm3'),
        ]
    
    # Guarda as posições dos obstáculos (assumindo estáticos)
    obstacles_xy = []
    for h in obstacle_handles:
        ox, oy, _ = sim.getObjectPosition(h, sim.handle_world)
        obstacles_xy.append((ox, oy))

    # Distância de segurança entre o robô e o obstaculo
    r_clear_env = rp.ROBOT_RADIUS + wp.OBSTACLE_RADIUS

    # Distância de segurança entre os robôs
    r_clear_rr = 2.0 * rp.ROBOT_RADIUS

    # listas para guardar trajetórias
    traj1 = []
    traj2 = []

    # pega a posição do goal uma vez (ele é estático)
    gx, gy, _ = sim.getObjectPosition(goal, sim.handle_world)

    # Definição dos sub-goals (vagas)
    d = 0.70  # distância entre as "vagas" (ajustar para ser mais separado)
    g1x, g1y = gx - d/2.0, gy     # sub-goal do robô 1
    g2x, g2y = gx + d/2.0, gy     # sub-goal do robô 2

    # flags de chegada
    reached1 = False
    reached2 = False

    # Variável para controlar o tempo
    start_time = time.time()

    while True:
        client.step()

        # Pose robô 1
        pose1 = sim.getObjectPose(robot1, sim.handle_world)
        r1x, r1y = pose1[0], pose1[1]
        yaw1 = yaw_from_quaternion(pose1[3], pose1[4], pose1[5], pose1[6])

        # Pose robô 2
        pose2 = sim.getObjectPose(robot2, sim.handle_world)
        r2x, r2y = pose2[0], pose2[1]
        yaw2 = yaw_from_quaternion(pose2[3], pose2[4], pose2[5], pose2[6])

        # log de trajetória
        traj1.append((r1x, r1y))
        traj2.append((r2x, r2y))

        # distâncias aos SUB-goals (para critério de chegada)
        dist_goal1 = math.hypot(g1x - r1x, g1y - r1y)
        dist_goal2 = math.hypot(g2x - r2x, g2y - r2y)

        # checa chegada individual: cada robô para quando entra no seu sub-goal
        if not reached1 and dist_goal1 <= rp.GOAL_TOL:
            reached1 = True
            sim.setJointTargetVelocity(leftMotor1, 0.0)
            sim.setJointTargetVelocity(rightMotor1, 0.0)
            print('Robô 1 chegou ao seu sub-goal.')

        if not reached2 and dist_goal2 <= rp.GOAL_TOL:
            reached2 = True
            sim.setJointTargetVelocity(leftMotor2, 0.0)
            sim.setJointTargetVelocity(rightMotor2, 0.0)
            print('Robô 2 chegou ao seu sub-goal.')

        # condição de parada: todos chegaram em seus sub-goals
        if reached1 and reached2:
            print('Todos os robôs chegaram aos seus sub-goals. Parando simulação.')
            break

        # Controle robô 1 (se ainda não chegou)
        if not reached1:
            # só considera robô 2 como obstáculo se ele ainda NÃO chegou
            other_pos_for_1 = (r2x, r2y) if not reached2 else None

            v1_cmd, w1_cmd, d_surf_min1 = compute_control_for_robot(
                r1x, r1y, yaw1,
                g1x, g1y,   # usa sub-goal 1
                obstacle_handles,
                sim,
                gains,
                rp,
                r_clear_env,
                other_robot_pos=other_pos_for_1,
                r_clear_rr=r_clear_rr
            )

            # reduzir v perto do seu sub-goal (baseado em GOAL_TOL, não GOAL_RADIUS)
            if dist_goal1 < 3.0 * rp.GOAL_TOL:
                v1_cmd *= 0.3

            # saturações
            v1_cmd = saturate(v1_cmd, -rp.V_MAX, rp.V_MAX)
            w1_cmd = saturate(w1_cmd, -rp.W_MAX, rp.W_MAX)

            wl1, wr1 = vw_to_wheel_omegas(v1_cmd, w1_cmd, rp.WHEEL_RADIUS, rp.AXLE_LENGTH)
            wl1 = saturate(wl1, -rp.WHEEL_OMEGA_MAX, rp.WHEEL_OMEGA_MAX) * rp.LEFT_SIGN
            wr1 = saturate(wr1, -rp.WHEEL_OMEGA_MAX, rp.WHEEL_OMEGA_MAX) * rp.RIGHT_SIGN

            sim.setJointTargetVelocity(leftMotor1, wl1)
            sim.setJointTargetVelocity(rightMotor1, wr1)

        # Controle robô 2 (se ainda não chegou)
        if not reached2:
            # só considera robô 1 como obstáculo se ele ainda NÃO chegou
            other_pos_for_2 = (r1x, r1y) if not reached1 else None

            v2_cmd, w2_cmd, d_surf_min2 = compute_control_for_robot(
                r2x, r2y, yaw2,
                g2x, g2y,   # usa sub-goal 2
                obstacle_handles,
                sim,
                gains,
                rp,
                r_clear_env,
                other_robot_pos=other_pos_for_2,
                r_clear_rr=r_clear_rr
            )

            if dist_goal2 < 3.0 * rp.GOAL_TOL:
                v2_cmd *= 0.3

            v2_cmd = saturate(v2_cmd, -rp.V_MAX, rp.V_MAX)
            w2_cmd = saturate(w2_cmd, -rp.W_MAX, rp.W_MAX)

            wl2, wr2 = vw_to_wheel_omegas(v2_cmd, w2_cmd, rp.WHEEL_RADIUS, rp.AXLE_LENGTH)
            wl2 = saturate(wl2, -rp.WHEEL_OMEGA_MAX, rp.WHEEL_OMEGA_MAX) * rp.LEFT_SIGN
            wr2 = saturate(wr2, -rp.WHEEL_OMEGA_MAX, rp.WHEEL_OMEGA_MAX) * rp.RIGHT_SIGN

            sim.setJointTargetVelocity(leftMotor2, wl2)
            sim.setJointTargetVelocity(rightMotor2, wr2)

        # Verificar tempo e interromper após 30 segundos
        elapsed_time = time.time() - start_time
        if elapsed_time > 30:
            print('Tempo limite alcançado. Finalizando simulação.')
            break

    # Encerrar simulação e zerar motores
    sim.setJointTargetVelocity(leftMotor1, 0.0)
    sim.setJointTargetVelocity(rightMotor1, 0.0)
    sim.setJointTargetVelocity(leftMotor2, 0.0)
    sim.setJointTargetVelocity(rightMotor2, 0.0)

    sim.stopSimulation()

    # Plotando as trajetórias
    if traj1 or traj2:
        fig, ax = plt.subplots()

        # obstáculos
        for (ox, oy) in obstacles_xy:
            # 1) Obstáculo físico
            circ_obs = plt.Circle((ox, oy),
                                  wp.OBSTACLE_RADIUS,
                                  fill=False,
                                  color='k')
            ax.add_patch(circ_obs)

            # 2) Região de influência do campo repulsivo
            raio_influencia = r_clear_env + gains.RHO_0
            circ_rep = plt.Circle((ox, oy),
                                  raio_influencia,
                                  fill=False,
                                  linestyle='--',
                                  alpha=0.5)
            ax.add_patch(circ_rep)

        # Trajetória robô 1
        if traj1:
            xs1 = [p[0] for p in traj1]
            ys1 = [p[1] for p in traj1]
            ax.plot(xs1, ys1, '-b', label='trajetória robô 1')
            ax.plot(xs1[0], ys1[0], 'bx', markersize=8, label='início robô 1')
            ax.plot(xs1[-1], ys1[-1], 'bo', markersize=6, label='fim robô 1')

        # Trajetória robô 2
        if traj2:
            xs2 = [p[0] for p in traj2]
            ys2 = [p[1] for p in traj2]
            ax.plot(xs2, ys2, '-r', label='trajetória robô 2')
            ax.plot(xs2[0], ys2[0], 'rx', markersize=8, label='início robô 2')
            ax.plot(xs2[-1], ys2[-1], 'ro', markersize=6, label='fim robô 2')

        # Goal (posição global)
        ax.plot(gx, gy, 'g*', markersize=14, label='goal (centro)')

        # Círculo do goal global (região onde cabem os robôs)
        goal_circle = plt.Circle((gx, gy),
                                 wp.GOAL_RADIUS,
                                 fill=False,
                                 linestyle='-',
                                 alpha=0.8)
        ax.add_patch(goal_circle)

        # Marcar sub-goals
        ax.plot(g1x, g1y, 'g^', markersize=8, label='sub-goal robô 1')
        ax.plot(g2x, g2y, 'gv', markersize=8, label='sub-goal robô 2')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Trajetórias dos Pioneers com repulsão mútua e sub-goals')

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
