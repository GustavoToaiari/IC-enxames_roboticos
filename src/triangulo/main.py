import math
import time
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import Robot

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)

# Ver formas de melhorar isso, para 3 robôs esta "ok". Mas, quando escalar para mais robôs, ficará inviável essa repetição de Robot.Robot
robots = [Robot.Robot(name= 'ePuck1',
              base=sim.getObject('/base1'),
              goal_path=sim.getObject('/Goal'),
              wheel_path=(sim.getObject('/leftJoint1'), sim.getObject('/rightJoint1')),
              sim=sim),
        Robot.Robot(name= 'ePuck2',
              base=sim.getObject('/base2'),
              goal_path=sim.getObject('/Goal'),
              wheel_path=(sim.getObject('/leftJoint2'), sim.getObject('/rightJoint2')),
              sim=sim),
        Robot.Robot(name= 'ePuck3',
              base=sim.getObject('/base3'),
              goal_path=sim.getObject('/Goal'),
              wheel_path=(sim.getObject('/leftJoint3'), sim.getObject('/rightJoint3')),
              sim=sim)]

if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()

start_sim = sim.getSimulationTime()

while True:
    client.step()

    arrived = [] # Guarda se cada robô chegou no goal
    for robot in robots:
         arrived.append(robot.run())

    if all(arrived): # Se todos os elementos da lista arrived forem verdadeiros, significa que todos chegaram no Goal
        for robot in robots:
            robot.stop_robot()
        print("Todos robôs chegaram no Goal.")
        sim.stopSimulation()
        break

    if sim.getSimulationTime() - start_sim > 60: # Critério de segurança, encerra se passar de 60 segundos de simulação
        for robot in robots:
            robot.stop_robot()
        print("Timeout. Encerrando.")
        sim.stopSimulation()
        break
