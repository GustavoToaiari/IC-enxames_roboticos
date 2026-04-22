import math
import time
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import Robot

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)

obstacles_handles = []
for i in range(1,15):
    obstacles_handles.append(sim.getObject('/80cmHighPillar25cm'+f"{i}"))

robots = []
for i in range(1,11):
    robots.append(Robot.Robot(name= 'ePuck'+f"{i}",
                base=sim.getObject('/base'+f"{i}"),
                goal_path=sim.getObject('/Goal'),
                wheel_path=(sim.getObject('/leftJoint'+f"{i}"), sim.getObject('/rightJoint'+f"{i}")),
                obstacles= obstacles_handles,
                sim=sim
                )
            )

if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()

start_sim = sim.getSimulationTime()

while True:
    client.step()

    arrived = [] # Guarda se cada robô chegou no goal
    for robot in robots:
        arrived.append(robot.run(robots))

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
