import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import Robot
import Parameters
import numpy as np

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)

created_epucks = Robot.Robot.create_epucks(sim, 4) # Vai criar 3 robos, pois o ePuck1 ja esta na cena
robots = []
for i in range(1, 5): # Vai percorrer 4 robos
    robots.append(Robot.Robot(name= 'ePuck'+f"{i}",
                base=sim.getObject('/ePuck'+f"{i}"+'/base'),
                goal_path=sim.getObject('/Goal'),
                wheel_path=(sim.getObject('/ePuck'+f"{i}"+'/leftJoint'), sim.getObject('/ePuck'+f"{i}"+'/rightJoint')),
                obstacles=[sim.getObject('/Cylinder1')],
                sim=sim))

if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()

start_sim = sim.getSimulationTime()
state = "FORM_TRIANGLE"
try:
    while True:
        client.step()

        if state == "FORM_TRIANGLE":
            max_errors = []

            for robot in robots:
                robot.run(robots, mode="formation")
                max_errors.append(robot.max_error)

            global_max_error = max(max_errors)
        
            if global_max_error < Parameters.DIST_TOL:
                for robot in robots:
                    robot.stop_robot()

                leader = min(robots, key=lambda r: np.linalg.norm(r.goal_position - r.position)) # Recebe robô e retorna o robô que tem a menor distância até o goal

                state = "GO_TO_GOAL"

        elif state == "GO_TO_GOAL":
            arrived = []

            for robot in robots:
                if robot is leader:
                    arrived.append(robot.run(robots, mode="go_to_goal"))
                
                else:
                    robot.run(robots, mode="formation")
                    arrived.append(False)
            
            if leader.arrived():
                for robot in robots:
                    robot.stop_robot()
                
                sim.stopSimulation()
                break


except KeyboardInterrupt:
    client_close = RemoteAPIClient()
    sim = client_close.getObject('sim')
    sim.stopSimulation()


Robot.Robot.remove_created_epucks(sim, created_epucks)