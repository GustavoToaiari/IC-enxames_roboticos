#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

@dataclass
class RobotParams:
    WHEEL_RADIUS: float = 0.0975   # [m]
    AXLE_LENGTH: float = 0.331     # [m]
    LEFT_SIGN: int = 1
    RIGHT_SIGN: int = 1
    V_MAX: float = 0.40            # [m/s]
    W_MAX: float = 1.2             # [rad/s]
    WHEEL_OMEGA_MAX: float = 12.0  # [rad/s]
    GOAL_TOL: float = 0.10         # [m]
    ROBOT_RADIUS: float = 0.20     # [m] raio "inflado" do Pioneer

@dataclass
class WorldParams:
    OBSTACLE_RADIUS: float = 0.25  # [m] raio físico do pilar (25 cm)
    OBSTACLE_PREFIX: str = '80cmHighPillar25cm'  # para auto-descobrir
    AUTO_DISCOVER: bool = True     # varrer cena por shapes com esse prefixo

@dataclass
class FieldGains:
    K_ATT: float = 1.0
    K_REP: float = 3.5
    RHO_0: float = 1.0            # [m] janela de influência SOBRE d_surf
    K_ROT: float = 0.7             # rotacional/tangencial p/ contornar

    K_V: float = 1.0               # mapeamento campo -> v
    K_W: float = 2.2               # mapeamento campo -> w

    K_W_EMERG: float = 2.5         # reforço de giro em emergência
    BACKUP_V: float = -0.05        # [m/s] recuo suave ao encostar

def yaw_from_quaternion(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

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

def repulsive_rotational_surface(px, py, ox, oy, gx, gy,
                                 K_REP, RHO0_surf, K_ROT,
                                 r_clear):
    """
    Campo repulsivo + rotacional usando distância à SUPERFÍCIE:
      d_surf = ||p - o|| - r_clear, com r_clear = r_robot + r_obs.
    Só atua se 0 < d_surf < RHO0_surf. Se d_surf <= 0, devolve empurrão forte.
    """
    dx, dy = px - ox, py - oy
    d = math.hypot(dx, dy)
    if d <= 1e-6:
        return 0.0, 0.0, -float('inf')  # caso patológico

    d_surf = d - r_clear
    ex, ey = dx / d, dy / d  # obs->robô unitário

    # sentido de giro: lado do objetivo relativo ao obstáculo
    ogx, ogy = gx - ox, gy - oy
    cross_sign = math.copysign(1.0, ex * ogy - ey * ogx)

    # encostou/entrou
    if d_surf <= 0.0:
        mag_rep = K_REP * (1.0 / max(1e-3, -d_surf))
        Fx_rep, Fy_rep = mag_rep * ex, mag_rep * ey
        tx, ty = -ey * cross_sign, ex * cross_sign
        Fx_rot, Fy_rot = K_ROT * tx, K_ROT * ty
        return Fx_rep + Fx_rot, Fy_rep + Fy_rot, d_surf

    # fora da influência
    if d_surf >= RHO0_surf:
        return 0.0, 0.0, d_surf

    # dentro da janela
    mag_rep = K_REP * (1.0/d_surf - 1.0/RHO0_surf) * (1.0/(d_surf*d_surf))
    Fx_rep, Fy_rep = mag_rep * ex, mag_rep * ey

    mag_rot = K_ROT * (1.0/d_surf - 1.0/RHO0_surf) * (1.0/(d_surf*d_surf))
    tx, ty = -ey * cross_sign, ex * cross_sign
    Fx_rot, Fy_rot = mag_rot * tx, mag_rot * ty

    return Fx_rep + Fx_rot, Fy_rep + Fy_rot, d_surf

def discover_obstacles(sim, prefix) -> list:
    """
    Retorna lista de handles de SHAPES cujo alias começa com `prefix`.
    Funciona em 4.10.00 e mantém fallback para variações de API.
    """
    handles = []

    # Tenta o caminho recomendado: toda a cena, apenas shapes
    try:
        shape_type = sim.object_shape_type  # constante oficial
    except AttributeError:
        shape_type = 4  # fallback típico para 'shape'

    try:
        # Preferível: SEMPRE retorna lista
        objs = sim.getObjectsInTree(sim.handle_scene, shape_type, 0)
    except Exception:
        # Fallback: algumas builds aceitam (type, options). Pode não retornar lista em todas.
        try:
            objs = sim.getObjects(shape_type, sim.handle_all)
        except TypeError:
            # Último fallback: versão antiga com 1 arg
            objs = sim.getObjects(shape_type)

    # Se por acaso vier um único inteiro, envelopa como lista
    if isinstance(objs, int):
        objs = [objs]
    # Algumas versões retornam tuple; normaliza para lista
    if not isinstance(objs, (list, tuple)):
        objs = list(objs)

    for h in objs:
        try:
            alias = sim.getObjectAlias(h, 0)
        except Exception:
            # Fallback caso a flag 0 não exista
            alias = sim.getObjectAlias(h)
        if isinstance(alias, bytes):
            alias = alias.decode('utf-8', errors='ignore')

        # Aceita tanto alias simples quanto caminho completo
        if alias.startswith(prefix) or alias.endswith('/' + prefix) or f'/{prefix}' in alias:
            handles.append(h)

    return handles



def main():
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    client.setStepping(True)

    robot = sim.getObject('/Pioneer_p3dx')
    leftMotor = sim.getObject('/Pioneer_p3dx_leftMotor')
    rightMotor = sim.getObject('/Pioneer_p3dx_rightMotor')
    goal = sim.getObject('/Goal')

    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()

    rp = RobotParams()
    wp = WorldParams()
    gains = FieldGains()

    # ---- Obstáculos ----
    if wp.AUTO_DISCOVER:
        obstacle_handles = discover_obstacles(sim, wp.OBSTACLE_PREFIX)
    else:
        # Especifique manualmente:
        obstacle_handles = [
            sim.getObject('/80cmHighPillar25cm0'),
            sim.getObject('/80cmHighPillar25cm1'),
            sim.getObject('/80cmHighPillar25cm2'),
            sim.getObject('/80cmHighPillar25cm3'),
            sim.getObject('/80cmHighPillar25cm4'),
            sim.getObject('/80cmHighPillar25cm5'),
            sim.getObject('/80cmHighPillar25cm6'),
            sim.getObject('/80cmHighPillar25cm7'),
            # ...
        ]

    if not obstacle_handles:
        print('[WARN] Nenhum obstáculo encontrado com o prefixo fornecido.')
    else:
        print(f'[INFO] Obstáculos detectados: {len(obstacle_handles)}')

    r_clear = rp.ROBOT_RADIUS + wp.OBSTACLE_RADIUS
    print(f'[INFO] Clearance geométrico: r_clear = {r_clear:.3f} m')

    while True:
        client.step()

        pose = sim.getObjectPose(robot, sim.handle_world)
        rx, ry = pose[0], pose[1]
        yaw = yaw_from_quaternion(pose[3], pose[4], pose[5], pose[6])

        gx, gy, _ = sim.getObjectPosition(goal, sim.handle_world)

        # Chegada ao goal
        dist_goal = math.hypot(gx - rx, gy - ry)
        if dist_goal <= rp.GOAL_TOL:
            sim.setJointTargetVelocity(leftMotor, 0.0)
            sim.setJointTargetVelocity(rightMotor, 0.0)
            print('[INFO] Meta atingida. Parando.')
            break

        # ---- Campos: atrativo + soma de todos os repulsivos/rotacionais ----
        Fx_att, Fy_att = attractive_field(rx, ry, gx, gy, gains.K_ATT)

        Fx_rep_sum, Fy_rep_sum = 0.0, 0.0
        d_surf_min = float('inf')

        for h in obstacle_handles:
            ox, oy, _ = sim.getObjectPosition(h, sim.handle_world)
            Fx_i, Fy_i, d_surf_i = repulsive_rotational_surface(
                rx, ry, ox, oy, gx, gy,
                gains.K_REP, gains.RHO_0, gains.K_ROT,
                r_clear
            )
            Fx_rep_sum += Fx_i
            Fy_rep_sum += Fy_i
            d_surf_min = min(d_surf_min, d_surf_i)

        Fx, Fy = Fx_att + Fx_rep_sum, Fy_att + Fy_rep_sum

        # vetor desejado (mundo) -> (v,w)
        vx_b, vy_b = world_vec_to_body(Fx, Fy, yaw)
        v_cmd = gains.K_V * math.hypot(vx_b, vy_b)
        w_cmd = gains.K_W * math.atan2(vy_b, max(1e-6, vx_b))

        # segurança: se encostou/entrou em QUALQUER obstáculo, acionar hard-stop
        if d_surf_min <= 0.0:
            v_cmd = min(0.0, gains.BACKUP_V)
            w_cmd = saturate(w_cmd * (gains.K_W_EMERG / max(1e-6, gains.K_W)),
                             -rp.W_MAX, rp.W_MAX)

        # (opcional) reduzir v perto do goal
        if dist_goal < 2.0 * rp.GOAL_TOL:
            v_cmd *= 0.3

        # saturações no corpo
        v_cmd = saturate(v_cmd, -rp.V_MAX, rp.V_MAX)
        w_cmd = saturate(w_cmd, -rp.W_MAX, rp.W_MAX)

        # corpo -> rodas
        wl, wr = vw_to_wheel_omegas(v_cmd, w_cmd, rp.WHEEL_RADIUS, rp.AXLE_LENGTH)
        wl = saturate(wl, -rp.WHEEL_OMEGA_MAX, rp.WHEEL_OMEGA_MAX) * rp.LEFT_SIGN
        wr = saturate(wr, -rp.WHEEL_OMEGA_MAX, rp.WHEEL_OMEGA_MAX) * rp.RIGHT_SIGN

        sim.setJointTargetVelocity(leftMotor, wl)
        sim.setJointTargetVelocity(rightMotor, wr)

    sim.stopSimulation()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('[ERRO]', e)
