from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
import numpy as np

# Conectar ao CoppeliaSIM
client = RemoteAPIClient()
sim = client.getObject('sim')
sim.setStepping(True)

# Obter o robô e os motores
robot = sim.getObject('/ePuck')
left_motor = sim.getObject('/ePuck/leftJoint')
right_motor = sim.getObject('/ePuck/rightJoint')
goal = sim.getObject('/Goal0')  # Ajuste para o nome do Goal

# Raio das rodas do ePuck
wheel_radius = 0.0425 / 2
L = 0.054  # Distância entre as rodas

# Verificar e iniciar a simulação
if sim.getSimulationState() != 0:
    sim.stopSimulation()
    time.sleep(2)

sim.startSimulation()
sim.step()

k_v = 0.2   # Ganho linear ajustado
k_w = 0.5   # Ganho angular ajustado

# Controlar o ePuck
while True:
    pos = sim.getObjectPosition(robot, sim.handle_world)    # Posição do robô
    ori = sim.getObjectOrientation(robot, sim.handle_world) # Orientação do robô
    theta = ori[2]  # Angulo yaw
    print(f"theta: ", theta)

    # Posição do Goal
    gx, gy, _ = sim.getObjectPosition(goal, sim.handle_world)

    # Cálculo da distância e erro angular
    dx = gx - pos[0]
    dy = gy - pos[1]
    distancia = math.sqrt(dx**2 + dy**2)

    if distancia < 0.05:  # Se o robô estiver suficientemente próximo, pare
        break

    angulo_desejado = math.atan2(dy, dx)
    erro_theta = angulo_desejado - theta
    erro_theta = math.atan2(math.sin(erro_theta), math.cos(erro_theta))  # Normaliza o erro de orientação

    # Calcular as velocidades
    v = k_v * distancia
    v = max(0.15, min(v, 0.4))  # Limitar a velocidade linear
    w = max(-1.0, min(k_w * erro_theta, 1.0))  # Limitar a velocidade angular

    # Velocidades das rodas
    v_r = (2 * v + w * L) / (2 * wheel_radius)
    v_l = (2 * v - w * L) / (2 * wheel_radius)

    # Limitar as velocidades das rodas
    max_wheel_speed = 6.28  # Defina o limite de velocidade para as rodas
    v_r = np.clip(v_r, -max_wheel_speed, max_wheel_speed)
    v_l = np.clip(v_l, -max_wheel_speed, max_wheel_speed)

    # Verificar as velocidades das rodas
    print(f"v_l: {v_l}, v_r: {v_r}")

    # Definir as velocidades das rodas
    sim.setJointTargetVelocity(left_motor, v_l)
    sim.setJointTargetVelocity(right_motor, v_r)

    # Verificar a posição do robô
    print(f"Posição do robô: {pos}")
    print(f"Posição do Goal: {gx}, {gy}")
    print(f"Distância: {distancia}")
    print(f"Erro de orientação (Theta): {erro_theta}")
    
    sim.step()
    time.sleep(0.01)

# Parar o robô no goal
sim.setJointTargetVelocity(left_motor, 0)
sim.setJointTargetVelocity(right_motor, 0)
print("Robô chegou ao goal!")

sim.stopSimulation()
