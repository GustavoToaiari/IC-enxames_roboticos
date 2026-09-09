import math
# Constantes
WHEEL_RADIUS = 0.0425 / 2
AXLE_LENGTH  = 0.054

V_MAX = 0.26 *4
W_MAX = V_MAX/WHEEL_RADIUS
WHEEL_OMEGA_MAX = 20.0

GOAL_TOL = 0.2
ROBOT_RADIUS = 0.0724/2

OBSTACLE_RADIUS = 0.25

# Ganhos do campo
K_ATT = 1
K_REP = 0.35 # Antes: 0.08
K_ROT = 50.0
REP_RANGE = 0.1

# Mapeamento força -> (v,w)
K_V = 0.2
K_W = 3.5 # Para 3 robôs: *6;

# Repulsão entre robôs
K_REP_ROBOTS = 0.35

# Formação triangular
DESIRED_DISTANCE = 0.30   # distância desejada entre cada par de robôs
DIST_TOL = 0.05          # tolerância da formação
V_MAX_FORMATION = 0.13*4 *4# Para 3 robôs: *4;

V_MAX_LEADER_GOAL = 0.8

# Formação em linha
LINE_DISTANCE = 0.2
LINE_TOL = 0.05
K_LINE = 1.5

LEADER_VISION_RADIUS = 0.5
MIN_PASSAGE_WIDTH = 0.6
LEADER_FOV_ANGLE = math.radians(120)


REP_SCALE_LEADER = 1.0
REP_SCALE_FOLLOWER = 0.05

F_REP_MAX_FOLLOWER = 0.25
F_REP_MAX_LEADER = 4.0

PASSAGE_TRANSITION_DELAY = 3.0  # segundos