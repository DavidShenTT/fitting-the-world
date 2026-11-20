"""Frozen settings for the real runs (PREREGISTRATION.md sections 2-9). Values marked TUNED are confirmed or set in
stage 2 on seeds 7000-7099 and recorded in PREREGISTRATION.md section 11 before any real seed runs."""

N_ITEMS = 600
DELTA, DRIFT_A, DRIFT_U = 0.0005, 0.05, 0.002
NOISE = 0.01                     # observation noise
BODY_TYPES = [1, 1, 1, 1, 2, 2, 2, 2]
LAM = {"unequal": 0.4, "symmetric": 0.4}   # TUNED (legibility, wild agents only)
LR = 0.05
EXPLORE = dict(eps_min=0.02, eps_m=0.6, m=0.2)   # TUNED (competence only); also the builders' fixed policy
BASE_WIDTH = 32

# builders
M_BUILD, BUILD_WIDTH, WARM = 6, 32, 30000
WEAR = dict(rule=1, eta=0.15)
EYE1 = dict(rule=2, s_u=0.3, kappa=1.0)
EYE025 = dict(rule=2, s_u=0.3, kappa=0.25)
NOSHAPE = dict(rule=0)
RULES = {"wear": WEAR, "eye1": EYE1, "eye025": EYE025, "none": NOSHAPE,
         "eye1_su01": dict(rule=2, s_u=0.1, kappa=1.0), "eye1_su003": dict(rule=2, s_u=0.03, kappa=1.0)}

# E1
E1_M = (2, 4, 6, 12)
E1_RULES = ("eye1", "eye025", "none")
E1_STEPS, E1_EVERY = 60000, 1000
STEADY_WINDOW, STEADY_TOL, STEADY_TAIL = 5000, 0.02, 10000

# E2 (symmetric, shaping by eye kappa = 1)
E2_SWEEP = dict(M=(1, 2, 4, 8, 16), s_u=(0.01, 0.03, 0.1, 0.3), delta=(0.0005, 0.001, 0.002, 0.004))
E2_RANGE = None                  # TUNED: chosen from the sweep by the rule in PREREGISTRATION.md section 5 (open)
# Code check only, never for real seeds: lets the smoke test run E2 end to end before the range is settled.
E2_SMOKE_RANGE = dict(M=(2, 4, 8), s_u=(0.3,), delta=(0.001, 0.002, 0.004), M_mid=4, s_u_mid=0.3, delta_mid=0.002)
E2_EXTEND = dict(M=(1, 2, 4, 8, 16), s_u=(0.003,), delta=(0.0005, 0.001, 0.002, 0.004, 0.008))
E2_EXTEND_D = dict(M=(1, 2, 4, 8, 16), s_u=(0.01, 0.03, 0.1, 0.3), delta=(0.008,))
E2_WIDTHS = (16, 32, 64)

# E3
E3_NEWBORNS_PER_MAP = 3
E3_CAP, E3_CHECK, E3_EARLY, E3_LL_AT, CODE_PAIRS = 20000, 250, 5000, 2000, 8000
COMP = 0.9

# E4-E6
RAISE_STEPS = 60000
E4_PAIRS = 5
ABANDON = 10000                  # 5 / delta

# E8-E10
WIDTH_LADDER = (4, 8, 16, 32, 64)  # narrowest legible width 4 (DEVIATIONS D1 yardstick) x (1, 2, 4, 8, 16)
T2 = 30000
PRETRAIN_CAP, PRETRAIN_EVERY, PLATEAU_TOL = 100000, 2000, 0.01
BUDGETS = (30000, 60000)
N_AG_GRID, N_AG_PRIMARY = 3, 4
EPS_MIN_ARMS = (0.1, 0.3)
EPS_R = {("wear", "skip"): 0.01, ("wear", "passive"): 0.02, ("eye1", "skip"): 0.01, ("eye1", "passive"): 0.02}
# TUNED on seeds 7000-7004 by home log loss only (results/tuning/epsr_summary.csv)
UPD_CAP = 240000
E9_STEPS, E9_EVERY = 500, 100
ENS_K, ENS_BETA = 5, 10.0
ENS_VARIANTS = {"standard": ([32] * 5, 0.0), "diverse": ([6, 13, 26, 45, 64], 0.0), "prior": ([32] * 5, 10.0)}
N_ENS = 3

REAL_SEEDS = tuple(range(10))
