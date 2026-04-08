import math
import time
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import Robot

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)

robot = Robot.Robot(name=sim.getObject('/ePuck'),
              base=sim.getObject('/base'),
              goal_path=sim.getObject('/Goal'),
              wheel_path=(sim.getObject('/leftJoint'), sim.getObject('/rightJoint')),
              sim=sim)

if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()

# Loop simulação
# start_time = time.time() # Funciona bem no meu computador pessoal | Usa o tempo real
start_sim = sim.getSimulationTime() # Funciona bem no computador do SIRO | Usa o tempo de simulação

while True:
    client.step()

    robot.run()

    if sim.getSimulationTime() - start_sim > 60: # Funciona bem no computador do SIRO
        print("Timeout. Encerrando.")
        break
