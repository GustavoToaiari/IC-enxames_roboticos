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

goal_paths = [
    sim.getObject('/Goal1'),
    sim.getObject('/Goal2'),
    sim.getObject('/Goal3')
]

obstacles = [
    sim.getObject('/Cuboid1'), sim.getObject('/Cuboid2'), sim.getObject('/Cuboid3'),
    sim.getObject('/Cuboid4'), sim.getObject('/Cuboid5'), sim.getObject('/Cuboid6'),
    sim.getObject('/Cuboid7'), sim.getObject('/Cuboid8'), sim.getObject('/Cuboid9'),
    sim.getObject('/Cuboid10'), sim.getObject('/Cuboid11'), sim.getObject('/Cuboid12')
]

robots = []
for i in range(1, 4): # Vai percorrer 4 robos
    robots.append(Robot.Robot(name= 'ePuck'+f"{i}",
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
current_goal_index = 1


def get_current_goal_name():
    return f"Goal{current_goal_index + 1}"

def get_current_target():
    return robots[0].goal_positions[current_goal_index]


def leader_arrived():
    return leader.arrived_target(get_current_target())


def run_leader_to_current_goal():
    leader.run(robots, mode="go_to_goal", target_position=get_current_target())

def clear_line_data():
    for robot in robots:
        robot.leader = None
        robot.line_order = None
        robot.line_target_position = None

def choose_leader_by_current_goal():
    global leader

    target = get_current_target()

    leader = min(robots, key=lambda r: np.linalg.norm(target -r.position))


def prepare_line_formation():
    global leader
    global line_order

    target = get_current_target()

    # O robô mais próximo do objetivo atual fica na frente
    line_order = sorted(
        robots,
        key=lambda r: np.linalg.norm(target - r.position)
    )

    leader = line_order[0]

    for robot in robots:
        robot.leader = leader
        robot.line_order = line_order
        robot.line_target_position = target.copy()


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
                clear_line_data()
                choose_leader_by_current_goal()

                print("Triângulo inicial formado.")
                print("Indo para o Goal1.")

                state = "MOVE_TRIANGLE"

        # MOVIMENTO EM TRIÂNGULO PARA O OBJETIVO ATUAL
        elif state == "MOVE_TRIANGLE":

            # Detecta passagem estreita
            if leader.detect_narrow_passage(robots):
                print(
                    f"Passagem estreita detectada indo para o {get_current_goal_name()}."
                )

                for robot in robots:
                    robot.stop_robot()

                time.sleep(2)

                prepare_line_formation()

                state = "FORM_LINE"
                continue

            # Movimento normal em triângulo
            for robot in robots:
                if robot is leader:
                    run_leader_to_current_goal()
                else:
                    robot.run(robots, mode="formation")

            # Verifica chegada ao objetivo atual
            if leader_arrived():
                for robot in robots:
                    robot.stop_robot()

                time.sleep(1)

                goal_reached = get_current_goal_name()
                print(f"{goal_reached} alcançado.")

                # Se ainda existem próximos goals
                if current_goal_index < len(goal_paths) - 1:
                    print("Formando triângulo novamente.")
                    state = "FORM_TRIANGLE_AFTER_GOAL"
                
                # Se chegou no último goal
                else:
                    print("Último objetivo alcançado. Encerrando simulação")
                    sim.stopSimulation()
                    break

        # FORMAÇÃO EM LINHA
        elif state == "FORM_LINE":
            max_errors = []

            for robot in robots:
                if robot is leader:
                    robot.stop_robot()
                    robot.max_error = 0.0
                else:
                    robot.run(
                        robots,
                        mode="line_formation"
                    )

                max_errors.append(robot.max_error)

            global_line_error = max(max_errors)

            if global_line_error < Parameters.LINE_TOL:
                for robot in robots:
                    robot.stop_robot()

                time.sleep(1)

                print(
                    f"Linha formada. Continuando para o {get_current_goal_name()}."
                )

                # Continua para o mesmo objetivo
                state = "MOVE_LINE"

        # MOVIMENTO EM LINHA PARA O OBJETIVO ATUAL
        elif state == "MOVE_LINE":

            for robot in robots:
                if robot is leader:
                    run_leader_to_current_goal()
                else:
                    robot.run(
                        robots,
                        mode="line_formation"
                    )

            if leader_arrived():
                for robot in robots:
                    robot.stop_robot()

                time.sleep(1)

                goal_reached = get_current_goal_name()
                print(f"{goal_reached} alcançado em formação de linha.")

                # Se ainda existem próximos goals
                if current_goal_index < len(goal_paths) - 1:
                    print("Formando triângulo novamente.")

                    state = "FORM_TRIANGLE_GOAL1"

                # Se chegou no úlimo goal
                else:
                    print("Último objetivo alcançado. Encerrado simulação")

                    sim.stopSimulation()
                    break

        # REFORMA O TRIÂNGULO NO GOAL1
        elif state == "FORM_TRIANGLE_GOAL1":
            max_errors = []

            for robot in robots:
                robot.run(robots, mode="formation")
                max_errors.append(robot.max_error)

            global_max_error = max(max_errors)

            if global_max_error < Parameters.DIST_TOL:
                for robot in robots:
                    robot.stop_robot()

                time.sleep(1)

                previous_goal = get_current_goal_name()

                # Avança para o próximo goal
                current_goal_index += 1

                clear_line_data()
                choose_leader_by_current_goal()

                print(f"Triângulo reformado após o {previous_goal}.")
                print(f"Indo para o {get_current_goal_name()}.")

                state = "MOVE_TRIANGLE"


except KeyboardInterrupt:
    client_close = RemoteAPIClient()
    sim = client_close.getObject('sim')
    sim.stopSimulation()


Robot.Robot.remove_created_epucks(
    sim,
    created_epucks
)