from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math

# Conectar ao CoppeliaSIM
client = RemoteAPIClient()
sim = client.getObject('sim')
sim.setStepping(True)

# Obter o robô e os motores (ajuste os caminhos conforme necessário)
robot = sim.getObject('/ePuck')
left_motor = sim.getObject('/ePuck/leftJoint')  # Motor esquerdo
right_motor = sim.getObject('/ePuck/rightJoint')  # Motor direito
wheel1 = sim.getObject('/ePuck/leftJoint/leftRespondableWheel')  # Roda esquerda
wheel2 = sim.getObject('/ePuck/rightJoint/rightRespondableWheel')  # Roda direita

# Raio das rodas do ePuck
wheel_radius = 0.033  # Ajuste conforme o valor correto para o ePuck
wheel_pos = sim.getObjectPosition(wheel1, sim.handle_world)
center_pos = sim.getObjectPosition(robot, sim.handle_world)
L = 0.1  # Distância entre as rodas do ePuck, ajustado conforme necessário

# Verificar e iniciar a simulação
if sim.getSimulationState() != 0:
    sim.stopSimulation()
    time.sleep(2)

sim.startSimulation()
sim.step()

# Definir os Goals (ajustar conforme necessário no seu cenário)
goal_names = ['/Goal']
goals = [sim.getObject(name) for name in goal_names]

k_v = 0.4   # Ganho linear
k_w = 0.8   # Ganho angular

# Controlar o ePuck para ir até cada Goal
for i, goal in enumerate(goals):
    print(f"Indo para o Goal {i+1}...")
    while True:
        pos = sim.getObjectPosition(robot, sim.handle_world)    # Posição [x, y, z]
        ori = sim.getObjectOrientation(robot, sim.handle_world) # Orientação [alpha, beta, gamma]
        theta = ori[2]  # yaw (rotação no plano XY)

        gx, gy, _ = sim.getObjectPosition(goal, sim.handle_world)   # Posição do Goal

        dx = gx - pos[0]
        dy = gy - pos[1]
        distancia = math.sqrt(dx**2 + dy**2)

        if distancia < 0.2:   # Para se a distância euclidiana for menor que 20 cm
            print(f"Goal {i+1} encontrado, distância final={distancia:.3f} m")
            break  # Sai do loop 'while'

        # Calcular o ângulo desejado para o movimento
        angulo_desejado = math.atan2(dy, dx)
        erro_theta = angulo_desejado - theta
        # Normalizar o erro de orientação para o intervalo [-pi, pi]
        erro_theta = math.atan2(math.sin(erro_theta), math.cos(erro_theta))

        v = k_v * distancia  # Velocidade linear proporcional à distância
        v = max(0.3, min(v, 0.8))  # Limitar a velocidade entre 0.3 e 0.8 m/s
        w = k_w * erro_theta  # Velocidade angular proporcional ao erro de orientação

        # Cálculo das velocidades das rodas
        v_r = (2*v + w*L) / (2*wheel_radius)
        v_l = (2*v - w*L) / (2*wheel_radius)

        # Definir as velocidades das rodas no CoppeliaSIM
        sim.setJointTargetVelocity(left_motor, v_l)
        sim.setJointTargetVelocity(right_motor, v_r)

        sim.step()

# Parar no Goal
sim.setJointTargetVelocity(left_motor, 0)
sim.setJointTargetVelocity(right_motor, 0)
print("Robô chegou ao goal!")

sim.stopSimulation()
