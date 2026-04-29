import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import Robot

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)

# robots = Robot.Robot.create_epucks_from_template(sim, 7) # Vai criar 5 robos, pois o ePuck1 ja esta na cena
robots = []
for i in range(1,11): # Vai percorrer 10 robos
    robots.append(Robot.Robot(name= 'ePuck'+f"{i}",
                base=sim.getObject('/ePuck'+f"{i}"+'/base'),
                goal_path=sim.getObject('/Goal'),
                wheel_path=(sim.getObject('/ePuck'+f"{i}"+'/leftJoint'), sim.getObject('/ePuck'+f"{i}"+'/rightJoint')),
                obstacles=[sim.getObject('/Cylinder1')],
                sim=sim))

if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()

start_sim = sim.getSimulationTime()

while True:
    client.step()

    arrived = [] # Guarda se cada robô chegou no goal
    for robot in robots:
        arrived.append(robot.run(robots))

    if robot.max_error < 0.04: # Verificar se esse 0.05 esta bugando algo (parece que esta fazendo uma formação diferente dependendo de onde os robôs estão inicialmente)
        for robot in robots:
            robot.stop_robot()

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