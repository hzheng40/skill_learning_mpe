"""
Default parameters for MPE environments

from JaxMARL: https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/environments/mpe/default_params.py
repackaged for simplicity
"""

# Action types
DISCRETE_ACT = "Discrete"
CONTINUOUS_ACT = "Continuous"

# Environment
CTF_MAX_STEPS = 200
MAX_STEPS = 25
# DAMPING = 0.25
DAMPING = 0.1
CONTACT_FORCE = 10
CONTACT_MARGIN = 1e-3
DT = 0.1

# Colours
AGENT_COLOUR = (115, 243, 115)
ADVERSARY_COLOUR = (243, 115, 115)
OBS_COLOUR = (64, 64, 64)

# Rendering
FLAG_PATH = [[0.0, 0.0], [0.0, 0.2], [0.1, 0.15], [0.0, 0.1]]