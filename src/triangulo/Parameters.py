# Constantes
WHEEL_RADIUS = 0.0425 / 2
AXLE_LENGTH  = 0.054

V_MAX = 0.13
W_MAX = V_MAX/WHEEL_RADIUS
WHEEL_OMEGA_MAX = 20.0

GOAL_TOL = 0.25
ROBOT_RADIUS = 0.0724/2

OBSTACLE_RADIUS = 0.25

# Ganhos do campo
K_ATT = 1
K_REP = 0.35 # Antes: 0.08
K_ROT = 50.0
REP_RANGE = 0.4

# Mapeamento força -> (v,w)
K_V = 0.2
K_W = 3.5 # Antes: 1.0

# Repulsão entre robôs
K_REP_ROBOTS = 0.35

# Formação triangular
DESIRED_DISTANCE = 0.50   # distância desejada entre cada par de robôs
DIST_TOL = 0.05          # tolerância da formação
