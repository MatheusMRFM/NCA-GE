"""
Constants
"""
TRAIN           = 0
TEST            = 1
REAL_NET        = 2

BETWEEN         = 0
CLOSE           = 1
EIGEN			= 2
CLUSTER_COEF    = 3
FLOW            = 4
SECOND_ORDER    = 5
HARMONIC        = 6
LOAD_CENT       = 7
KATZ            = 8

RANK            = 0
VALUE           = 1

FLD_GRAPH       = "../Graphs/"
TRAIN_SUB_FLD   = "Train/"
#TRAIN_SUB_FLD   = "Train_Mix/"
TEST_SUB_FLD    = "Test/"

"""
General variables
"""
LOAD            = False
MODE            = REAL_NET
CENTRALITY      = BETWEEN
TARGET          = RANK
INPUT           = RANK
NUM_WORKER      = 1
TRAIN_LENGHT    = 1
SAVE_NETWORK    = True
SAVER_INTERVAL  = 10
USE_GPU         = True
MAX_STEPS       = 1.5e6
TEST_NUM        = 0

"""
Training variables
"""
N_PARALLEL_BATCH= 50
BATCH_SIZE      = 256
TH_CENT         = -1.0
K_FIRST         = 0.3

"""
Neural Network variables
"""
UNITS_H1        = 20
UNITS_H2        = 20
UNITS_H3        = 20
LAMBDA_REGUL    = 1e-3
LEARNING_RATE   = 1e-2
INIT_WEIGHT     = 0.1

"""
Summary variables
"""
SUMMARY_INTERVAL    = 500
SUMMARY_NAME        = "{:.0e}".format(LEARNING_RATE) + "_WEIGHT_" + str(INIT_WEIGHT)
MODEL_PATH      	= "./Model/" + SUMMARY_NAME + "/"
