import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from Robot import Robot
import Parameters
import numpy as np
import time

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)

created_epucks = Robot.create_epucks(sim, 3) # Vai criar 3 robos, pois o ePuck1 ja esta na cena

goal_paths = [
    sim.getObject('/Goal1'),
    sim.getObject('/Goal2'),
    sim.getObject('/Goal3'),
    sim.getObject('/Goal4')
]

obstacles = [
    sim.getObject('/Cuboid1'), sim.getObject('/Cuboid2'), sim.getObject('/Cuboid3'),
    sim.getObject('/Cuboid4'), sim.getObject('/Cuboid5'), sim.getObject('/Cuboid6'),
    sim.getObject('/Cuboid7'), sim.getObject('/Cuboid8'), sim.getObject('/Cuboid9'),
    sim.getObject('/Cuboid10'), sim.getObject('/Cuboid11'), sim.getObject('/Cuboid12'),
    sim.getObject('/Cylinder1')
]

robots = []
for i in range(1, 4): # Vai percorrer 4 robos
    robots.append(Robot(name= 'ePuck'+f"{i}",
                base=sim.getObject('/ePuck'+f"{i}"+'/base'),
                goal_paths=goal_paths,
                wheel_path=(sim.getObject('/ePuck'+f"{i}"+'/leftJoint'), sim.getObject('/ePuck'+f"{i}"+'/rightJoint')),
                obstacles=obstacles,
                sim=sim))

if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()

start_sim = sim.getSimulationTime()
state = "FORM_TRIANGLE_INITIAL"

leader = None
line_order = None

# Objetivo atual:
# 1 = Goal1
# 2 = Goal2
# 3 = Goal2
current_goal_index = 0


try:
    while True:
        client.step()

        for robot in robots:
            robot.get_pose_2d()

        # FORMAÇÃO TRIANGULAR INICIAL
        if state == "FORM_TRIANGLE_INITIAL":
            max_errors = []

            for robot in robots:
                robot.run(robots, mode="formation")
                max_errors.append(robot.max_error)

            global_max_error = max(max_errors)

            if global_max_error < Parameters.DIST_TOL:
                for robot in robots:
                    robot.stop_robot()

                time.sleep(1)

                current_goal_index = 0
                target_position = robots[0].goal_positions[current_goal_index]

                Robot.clear_line_data(robots)
                leader = Robot.choose_leader(robots, target_position)

                state = "MOVE_TRIANGLE"

        # MOVIMENTO EM TRIÂNGULO PARA O OBJETIVO ATUAL
        elif state == "MOVE_TRIANGLE":

            # Detecta passagem estreita
            if leader.detect_narrow_passage(target_position):
                leader, line_order = Robot.prepare_line_formation(robots, target_position)

                state = "MOVE_LINE"
                continue

            # Movimento normal em triângulo
            for robot in robots:
                if robot is leader:
                    robot.run(robots, mode="go_to_goal", target_position=target_position)
                else:
                    robot.run(robots, mode="formation")

            # Verifica chegada ao objetivo atual
            if leader.arrived_target(target_position):
                for robot in robots:
                    robot.stop_robot()

                time.sleep(1)

                # Se ainda existem próximos goals
                if current_goal_index < len(goal_paths) - 1:
                    state = "FORM_TRIANGLE_AFTER_GOAL"
                
                # Se chegou no último goal
                else:
                    time.sleep(2)
                    sim.stopSimulation()
                    break

        # FORMAÇÃO EM LINHA
        elif state == "FORM_LINE":
            line_ready = Robot.form_line(robots, leader)

            if line_ready:
                for robot in robots:
                    robot.stop_robot()
                time.sleep(1)

                state = "MOVE_LINE"

        # MOVIMENTO EM LINHA PARA O OBJETIVO ATUAL
        elif state == "MOVE_LINE":


            # ============================================
            # Verifica se o último robô saiu da passagem
            # ============================================

            last_robot = Robot.get_last_robot_line(line_order)


            if last_robot.detect_narrow_passage(target_position):

                Robot.clear_line_data(robots)


                # Escolhe novamente o robô mais próximo
                # do objetivo como líder
                leader = Robot.choose_leader(
                    robots,
                    target_position
                )


                state = "FORM_TRIANGLE_AFTER_PASSAGE"

                continue



            # ============================================
            # Movimento normal em linha
            # ============================================

            for robot in robots:

                if robot is leader:

                    robot.run(
                        robots,
                        mode="go_to_goal",
                        target_position=target_position
                    )

                else:

                    robot.run(
                        robots,
                        mode="line_formation"
                    )



            # ============================================
            # Chegou no objetivo
            # ============================================

            if leader.arrived_target(target_position):

                for robot in robots:
                    robot.stop_robot()


                time.sleep(1)


                if current_goal_index < len(goal_paths)-1:

                    state = "FORM_TRIANGLE_AFTER_GOAL"

                else:

                    sim.stopSimulation()
                    break

                # ============================================
        # Retorno da linha para triângulo após passagem
        # ============================================

        elif state == "FORM_TRIANGLE_AFTER_PASSAGE":


            max_errors = []


            for robot in robots:


                robot.run(
                    robots,
                    mode="formation_with_goal",
                    target_position=target_position
                )


                max_errors.append(robot.max_error)



            # Formação triangular concluída

            if max(max_errors) < Parameters.DIST_TOL:


                for robot in robots:
                    robot.stop_robot()


                time.sleep(0.5)


                Robot.clear_line_data(robots)


                # Novo líder
                leader = Robot.choose_leader(
                    robots,
                    target_position
                )


                state = "MOVE_TRIANGLE"

        # REFORMA O TRIÂNGULO NO GOAL
        elif state == "FORM_TRIANGLE_AFTER_GOAL":
            triangle_ready = Robot.form_triangle(robots)


            current_goal_index += 1
            target_position = robots[0].goal_positions[current_goal_index]

            Robot.clear_line_data(robots)
            leader = Robot.choose_leader(robots, target_position)

            state = "MOVE_TRIANGLE"


except KeyboardInterrupt:
    client_close = RemoteAPIClient()
    sim = client_close.getObject('sim')
    sim.stopSimulation()


Robot.remove_created_epucks(
    sim,
    created_epucks
)