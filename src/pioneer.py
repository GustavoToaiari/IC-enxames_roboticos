import math
import time
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# Constantes
WHEEL_RADIUS = 0.0975
AXLE_LENGTH  = 0.331
LEFT_SIGN    = 1
RIGHT_SIGN   = 1

V_MAX = 0.8
W_MAX = 2.4
WHEEL_OMEGA_MAX = 12.0

GOAL_TOL = 0.10
ROBOT_RADIUS = 0.20

OBSTACLE_RADIUS = 0.50

# Retângulo de sub-goals no goal
GOAL_WIDTH  = 18.0
GOAL_HEIGHT = 1.0

# Ganhos do campo
K_ATT = 1.0
K_REP = 10.0
RHO_0 = 1.0
K_ROT = 20.0

# Mapeamento força -> (v,w)
K_V = 1.0
K_W = 2.2



# Funções pequenas
def yaw_from_quaternion(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw*qz + qx*qy), 1.0 - 2.0 * (qy*qy + qz*qz))

def saturate(x, a, b):
    return max(min(x, b), a)

def world_to_body(vx_w, vy_w, yaw):
    # gira o vetor do mundo para o corpo (frame do robô)
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    vx_b = c*vx_w - s*vy_w
    vy_b = s*vx_w + c*vy_w
    return vx_b, vy_b

def vw_to_wheels(v, w, r, L):
    wr = (2.0*v + w*L) / (2.0*r)
    wl = (2.0*v - w*L) / (2.0*r)
    return wl, wr

def repulsive_rotational_surface(px, py, ox, oy, gx, gy, r_clear):
    # retorna Fx, Fy e d_surf (distância até "superfície" do obstáculo)
    dx = px - ox
    dy = py - oy
    d  = math.hypot(dx, dy)

    # evita divisão por zero se cair em cima
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
    if d_surf >= RHO_0:
        return 0.0, 0.0, d_surf

    # dentro da influência
    mag = (1.0/d_surf - 1.0/RHO_0) * (1.0/(d_surf*d_surf))

    mag_rep = K_REP * mag
    Fx_rep = mag_rep * ex
    Fy_rep = mag_rep * ey

    mag_rot = K_ROT * mag
    tx = -ey * cross_sign
    ty =  ex * cross_sign
    Fx_rot = mag_rot * tx
    Fy_rot = mag_rot * ty

    return Fx_rep + Fx_rot, Fy_rep + Fy_rot, d_surf


# Principal
def main():
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    client.setStepping(True)

    N = 10  # Número de robôs

    # Handles dos robôs
    robots = []
    for i in range(N):
        suffix = '' if i == 0 else str(i + 1)

        rob_h = sim.getObject('/Pioneer_p3dx' + suffix)
        lm_h  = sim.getObject('/Pioneer_p3dx_leftMotor' + suffix)
        rm_h  = sim.getObject('/Pioneer_p3dx_rightMotor' + suffix)

        robots.append({
            'rob': rob_h,
            'lm': lm_h,
            'rm': rm_h,
            'traj': [],
            'reached': False,
            'pose': (0.0, 0.0, 0.0),
            'subgoal': (0.0, 0.0)
        })

    goal_h = sim.getObject('/Goal')

    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()

    # Obstáculos
    obstacle_handles = [
        sim.getObject('/80cmHighPillar25cm0'),
        sim.getObject('/80cmHighPillar25cm1'),
        sim.getObject('/80cmHighPillar25cm2'),
        sim.getObject('/80cmHighPillar25cm3'),
    ]

    obstacles_xy = []
    for h in obstacle_handles:
        ox, oy, _ = sim.getObjectPosition(h, sim.handle_world)
        obstacles_xy.append((ox, oy))

    # distâncias de segurança
    r_clear_env = ROBOT_RADIUS + OBSTACLE_RADIUS
    r_clear_rr  = 2.0 * ROBOT_RADIUS

    # Goal e subgoals
    gx, gy, _ = sim.getObjectPosition(goal_h, sim.handle_world)

    spacing_x = GOAL_WIDTH / (N - 1) if N > 1 else 0.0
    for k in range(N):
        sx = gx + (k * spacing_x) - GOAL_WIDTH/2
        sy = gy
        robots[k]['subgoal'] = (sx, sy)

    # Loop simulação
    start_time = time.time()

    while True:
        client.step()

        # 1) Lê pose e salva trajetória
        for rob in robots:
            pose = sim.getObjectPose(rob['rob'], sim.handle_world)
            x, y = pose[0], pose[1]
            yaw = yaw_from_quaternion(pose[3], pose[4], pose[5], pose[6])
            rob['pose'] = (x, y, yaw)
            rob['traj'].append((x, y))

        # 2) Checa se algum robô chegou ao sub-goal
        for rob in robots:
            if rob['reached']:
                continue
            sx, sy = rob['subgoal']
            x, y, _ = rob['pose']

            # Se chegou, para o robô e marca como "reached"
            if math.hypot(sx - x, sy - y) <= GOAL_TOL:
                rob['reached'] = True
                sim.setJointTargetVelocity(rob['lm'], 0.0)
                sim.setJointTargetVelocity(rob['rm'], 0.0)
                print("Um robô chegou ao sub-goal!")

        # 3) Se todos chegaram, encerra
        if all(r['reached'] for r in robots):
            print("Todos chegaram. Encerrando.")
            break

        # 4) Controle (um por um)
        for i, rob in enumerate(robots):
            if rob['reached']:
                continue

            x, y, yaw = rob['pose']
            sx, sy = rob['subgoal']

            # Campo atrativo
            Fx = K_ATT * (sx - x)
            Fy = K_ATT * (sy - y)

            # Repulsão + rotacional dos pilares
            for h in obstacle_handles:
                ox, oy, _ = sim.getObjectPosition(h, sim.handle_world)
                fxi, fyi, _ = repulsive_rotational_surface(x, y, ox, oy, sx, sy, r_clear_env)
                Fx += fxi
                Fy += fyi

            # Outros robôs como obstáculos móveis
            for j, other in enumerate(robots):
                if j == i:
                    continue
                if other['reached']:
                    continue
                oxr, oyr, _ = other['pose']
                fxi, fyi, _ = repulsive_rotational_surface(x, y, oxr, oyr, sx, sy, r_clear_rr)
                Fx += fxi
                Fy += fyi

            # Força (mundo) -> comando (v,w)
            vx_b, vy_b = world_to_body(Fx, Fy, yaw)

            v_cmd = K_V * math.hypot(vx_b, vy_b)
            w_cmd = K_W * math.atan2(vy_b, max(1e-6, vx_b))

            # desacelera perto do sub-goal
            dist_goal = math.hypot(sx - x, sy - y)
            if dist_goal < 2.0 * GOAL_TOL:
                v_cmd *= 0.4

            # saturações
            v_cmd = saturate(v_cmd, -V_MAX, V_MAX)
            w_cmd = saturate(w_cmd, -W_MAX, W_MAX)

            wl, wr = vw_to_wheels(v_cmd, w_cmd, WHEEL_RADIUS, AXLE_LENGTH)
            wl = saturate(wl, -WHEEL_OMEGA_MAX, WHEEL_OMEGA_MAX) * LEFT_SIGN
            wr = saturate(wr, -WHEEL_OMEGA_MAX, WHEEL_OMEGA_MAX) * RIGHT_SIGN

            sim.setJointTargetVelocity(rob['lm'], wl)
            sim.setJointTargetVelocity(rob['rm'], wr)

        # 5) timeout
        if time.time() - start_time > 120:
            print("Timeout. Encerrando.")
            break

    # Para tudo e plota
    for rob in robots:
        sim.setJointTargetVelocity(rob['lm'], 0.0)
        sim.setJointTargetVelocity(rob['rm'], 0.0)

    sim.stopSimulation()

    fig, ax = plt.subplots()

    # obstáculos + raio influência
    for (ox, oy) in obstacles_xy:
        ax.add_patch(plt.Circle((ox, oy), OBSTACLE_RADIUS, fill=False, color='k'))
        ax.add_patch(plt.Circle((ox, oy), r_clear_env + RHO_0, fill=False, linestyle='--', alpha=0.5))

    # trajetórias
    for idx, rob in enumerate(robots):
        if not rob['traj']:
            continue
        xs = [p[0] for p in rob['traj']]
        ys = [p[1] for p in rob['traj']]
        ax.plot(xs, ys, label=f'robô {idx+1}')
        ax.plot(xs[0], ys[0], 'x', markersize=6)
        ax.plot(xs[-1], ys[-1], 'o', markersize=5)

    # goal (centro) + retângulo
    ax.plot(gx, gy, '*', markersize=14, label='goal (centro)')
    ax.add_patch(plt.Rectangle((gx - GOAL_WIDTH/2, gy - GOAL_HEIGHT/2),
                               GOAL_WIDTH, GOAL_HEIGHT,
                               fill=False, linestyle='-', alpha=0.8))

    # subgoals
    for k, rob in enumerate(robots):
        sx, sy = rob['subgoal']
        ax.plot(sx, sy, '^', markersize=7)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Trajetórias de {N} Pioneers (campo potencial + repulsão mútua)')
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
