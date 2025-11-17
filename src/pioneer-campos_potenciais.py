from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import math
import time

# Parâmetros
K_atracao = 1.0 # ganho de atração
K_repulsao = 1.0    # ganho de repulsão
R_obstaculo = 0.25  # raio do obstáculo (pilar de 25cm)
RHO_0 = 1.0 # 0.2m além do obstáculo
V_max = 0.8
W_max = 1.0

client = RemoteAPIClient()
sim = client.getObject('sim')
sim.setStepping(True)

robot = sim.getObject('/Pioneer_p3dx')
left_motor = sim.getObject('/Pioneer_p3dx_leftMotor')
right_motor = sim.getObject('/Pioneer_p3dx_rightMotor')

if sim.getSimulationState() != 0:
    sim.stopSimulation()
    time.sleep(2)

sim.startSimulation()
sim.step()

# Goal e Obstáculo únicos
goal = sim.getObject('/Goal[0]')
obstaculo = sim.getObject('/80cmHighPillar25cm')

# Posições fixas (no mundo)
goal_pos = np.array(sim.getObjectPosition(goal, sim.handle_world))[:2]
obstaculo_pos = np.array(sim.getObjectPosition(obstaculo, sim.handle_world))[:2]

# Funções auxiliares
def get_position():
    pos = sim.getObjectPosition(robot, -1)
    return np.array([pos[0], pos[1]])

def get_orientation():
    euler = sim.getObjectOrientation(robot, -1)
    return euler[2]  # yaw

def attractive_force(x, goal):
    return - K_atracao * (x - goal)

def repulsive_force(x, obs, obs_radius=0.25, robot_radius=0.19): # raio do robô aproximado
    diff = x - obs
    dist_center = np.linalg.norm(diff)
    rho = dist_center - (obs_radius + robot_radius)  # distância entre superfícies

    if rho <= RHO_0 and rho > 1:
        direction = diff / dist_center
        return K_repulsao * (1/rho - 1/RHO_0) * (1/(rho**2)) * direction
    elif rho <= 1:
        # Dentro da zona de colisão —> empurra com força máxima
        direction = diff / (dist_center + 1e-6)
        return K_repulsao * (1/(1e-3)) * direction
    else:
        return np.zeros(2)

def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

def control_to_wheels(v, w):
    L = 0.381  # distância entre rodas
    R = 0.0975  # raio da roda
    vL = (v - w * L / 2) / R
    vR = (v + w * L / 2) / R
    return vL, vR

while True:
    sim.step()

    x = get_position()
    theta = get_orientation()

    F_att = attractive_force(x, goal_pos)
    F_rep = repulsive_force(x, obstaculo_pos, obs_radius=0.25)
    F_total = F_att + F_rep

    desired_angle = math.atan2(F_total[1], F_total[0])
    angle_error = normalize_angle(desired_angle - theta)

    # velocidade proporcional a força total
    v = min(V_max, np.linalg.norm(F_total))
    w = 1.5 * angle_error

    vL, vR = control_to_wheels(v, w)
    sim.setJointTargetVelocity(left_motor, vL)
    sim.setJointTargetVelocity(right_motor, vR)

    if np.linalg.norm(goal_pos - x) < 0.2:
        sim.setJointTargetVelocity(left_motor, 0)
        sim.setJointTargetVelocity(right_motor, 0)
        print("Goal atingido!")
        break

    time.sleep(0.05)

sim.setJointTargetVelocity(left_motor, 0)
sim.setJointTargetVelocity(right_motor, 0)
print("Robô chegou ao goal!")

sim.stopSimulation()