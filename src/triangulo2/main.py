import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import Robot
import Parameters
import numpy as np
import time

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)

created_epucks = Robot.Robot.create_epucks(sim, 3) # Vai criar 3 robos, pois o ePuck1 ja esta na cena
robots = []
for i in range(1, 4): # Vai percorrer 4 robos
    robots.append(Robot.Robot(name= 'ePuck'+f"{i}",
                base=sim.getObject('/ePuck'+f"{i}"+'/base'),
                goal_path=sim.getObject('/Goal1'),
                goal2_path=sim.getObject('/Goal2'),
                wheel_path=(sim.getObject('/ePuck'+f"{i}"+'/leftJoint'), sim.getObject('/ePuck'+f"{i}"+'/rightJoint')),
                obstacles=[sim.getObject('/Cylinder1')],
                sim=sim))

if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()

start_sim = sim.getSimulationTime()
state = "FORM_TRIANGLE"
leader = None
line_order = None
try:
    while True:
        client.step()

        for robot in robots:
            robot.get_pose_2d()

        if state == "FORM_TRIANGLE":
            max_errors = []

            for robot in robots:
                robot.run(robots, mode="formation")
                max_errors.append(robot.max_error)

            global_max_error = max(max_errors)
        
            if global_max_error < Parameters.DIST_TOL:
                for robot in robots:
                    robot.stop_robot()

                time.sleep(1)

                # Escolhe o robô mais próximo do Goal1 como líder
                leader = min(robots, key=lambda r: np.linalg.norm(r.goal_position - r.position)) # Recebe robô e retorna o robô que tem a menor distância até o goal

                state = "GO_TO_GOAL"

        elif state == "GO_TO_GOAL":

            for robot in robots:
                if robot is leader:
                    robot.run(robots, mode="go_to_goal")
                
                else:
                    robot.run(robots, mode="formation")
            
            if leader.arrived():
                for robot in robots:
                    robot.stop_robot()

                time.sleep(1)

                # Ordena os robôs pela distância até o Goal2
                # O mais proximo do Goal2 fica na frente da fila
                line_order = sorted(robots, key=lambda r: np.linalg.norm(r.goal2_position - r.position))

                # O lider para o Goal2 é o primeiro da fila
                leader = line_order[0]

                # Informa a todos o todos quem é o lider e a ordem da linha
                for robot in robots:
                    robot.leader = leader
                    robot.line_order = line_order
                
                state = "FORM_LINE"
        
        elif state == "FORM_LINE":
            max_errors = []

            for robot in robots:
                if robot is leader:
                    robot.stop_robot()
                    robot.max_error = 0
                else:
                    robot.run(robots, mode="line_formation")
                
                max_errors.append(robot.max_error)
            
            global_line_error = max(max_errors)

            if global_line_error < Parameters.LINE_TOL:
                for robot in robots:
                    robot.stop_robot()

                time.sleep(1)
                state = "GO_TO_GOAL2"

        elif state == "GO_TO_GOAL2":            
            for robot in robots:
                if robot is leader:
                    robot.run(robots, mode="go_to_goal2")
                else:
                    robot.run(robots, mode="line_formation")
            
            if leader.arrived2():
                for robot in robots:
                    robot.stop_robot()
                
                sim.stopSimulation()
                break


except KeyboardInterrupt:
    client_close = RemoteAPIClient()
    sim = client_close.getObject('sim')
    sim.stopSimulation()


Robot.Robot.remove_created_epucks(sim, created_epucks)