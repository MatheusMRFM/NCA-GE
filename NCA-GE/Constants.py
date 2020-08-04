"""
Constants
"""
TRAIN           = 0
TEST            = 1
REAL_NET        = 2

BETWEEN         = 0
CLOSE           = 1
EIGEN           = 2
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
MODE            = TRAIN    #REAL_NET
CENTRALITY      = HARMONIC

NUM_WORKER      = 1
SAVE_NETWORK    = True
SAVER_INTERVAL  = 25
USE_GPU         = True

MAX_STEPS       = 7e6
TEST_NUM        = 0

K_FIRST         = 0.3

"""
Embedding variables
"""
NUM_FEATURES    = 1
EMBED_SIZE      = 512
USE_NORMALIZE   = False

RANGE_01        = 0
RANGE_11        = 1
RANGE           = RANGE_01

INPUT           = RANK
TARGET          = RANK

TH_CENT         = 0.0

if RANGE == RANGE_11:
    TH_CENT = 2.0 * TH_CENT - 1.0   #to [-1, 1]

"""
Neural Network variables
"""
UNITS_H1        = int(EMBED_SIZE/2)
UNITS_H2        = int(EMBED_SIZE/2)
INIT_WEIGHT     = 1e-1

LAMBDA_REGUL    = 1e-3
LEARNING_RATE   = 1e-3
MIN_LEARNING_RT	= 1e-4
DECAY           = 0.999
BATCH_SIZE      = 1024 #128   #64    #16    #16    #32

GCN             = 0
S2VEC           = 1
EMBED_METHOD    = GCN

LAYERS          = 2

"""
Summary variables
"""
NORM_NAME = ''
if USE_NORMALIZE:
    NORM_NAME = '_NORM'

IN_RANK_NAME = 'RANK'
if INPUT == VALUE:
    IN_RANK_NAME = 'VALUE'

OUT_RANK_NAME = 'RANK'
if TARGET == VALUE:
    OUT_RANK_NAME = 'VALUE'

METHOD_NAME = 'GCN'
if EMBED_METHOD == S2VEC:
    METHOD_NAME = 'S2VEC'

RANGE_NAME = '01'
if RANGE == RANGE_11:
    RANGE_NAME = '11'

"""
Summary variables
"""
SUMMARY_INTERVAL    = 500
SUMMARY_NAME        =   METHOD_NAME + NORM_NAME + \
                        "_LAYER=" + str(LAYERS) + \
                        "_LR=" + "{:.0e}".format(LEARNING_RATE) + "-" + "{:.0e}".format(MIN_LEARNING_RT) + \
                        "_F=" + str(NUM_FEATURES) + \
                        "_IN=" + IN_RANK_NAME + \
                        "_OUT=" + OUT_RANK_NAME + \
                        "_RANGE=" + RANGE_NAME + \
                        "_BATCH=" + str(BATCH_SIZE) + \
                        "_EMBED=" + str(EMBED_SIZE) + \
                        "_LAMBDA=" + str(LAMBDA_REGUL) + \
                        "_TH=" + "{:.2f}".format(TH_CENT) + \
                        "_" + str(TEST_NUM)

MODEL_PATH      = "./Model/" + SUMMARY_NAME + '/'
