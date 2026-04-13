import math
import time
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import Robot

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)

robot = Robot.Robot(name= 'ePuck',
              base=sim.getObject('/base'),
              goal_path=sim.getObject('/Goal'),
              wheel_path=(sim.getObject('/leftJoint'), sim.getObject('/rightJoint')),
              sim=sim)

if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()

start_sim = sim.getSimulationTime()

while True:
    client.step()

    arrived = robot.run()

    if arrived:
        robot.stop_robot()
        print("Robô chegou no goal.")
        sim.stopSimulation()
        break

    if sim.getSimulationTime() - start_sim > 60:
        robot.stop_robot()
        print("Timeout. Encerrando.")
        sim.stopSimulation()
        break
