DJ_DEBUG = 1

# GA Setup
POPSIZE = 100

STOP_CONDITION = "iter"
#STOP_CONDITION = "time"

NGEN = 300
RUNTIME = 3600
STEPSIZE = 10
# Mutation Hyperparameters
#MUTATION_TYPE = "random"
MUTATION_TYPE = "attention-based"
# range of the mutation
MUTLOWERBOUND = 0.01
MUTUPPERBOUND = 0.6

# Reseeding Hyperparameters
# extent of the reseeding operator
RESEEDUPPERBOUND = 10

K_SD = 0.1

# K-nearest
K = 1

# Archive configuration
ARCHIVE_THRESHOLD = 4.0

#------- NOT TUNING ----------

# mutation operator probability
MUTOPPROB = 0.5
MUTOFPROB = 0.5

IMG_SIZE = 28
num_classes = 10


INITIALPOP = 'seeded'

GENERATE_ONE_ONLY = False

MODEL2 = 'models/cnnClassifier_lowLR.h5'
MODEL = 'models/cnnClassifier.h5'
#MODEL = "models/regular3"
#MODEL = 'models/cnnClassifier_001.h5'
#MODEL = 'models/cnnClassifier_op.h5'

RESULTS_PATH = 'results'
REPORT_NAME = 'stats.csv'
DATASET = 'original_dataset/janus_dataset_comparison.h5'
EXPLABEL = 5

#TODO: set interpreter
INTERPRETER = '/home/vin/yes/envs/tf_gpu/bin/python'

#Attention Maps Options:

MUTANTS_ROOT_FOLDER = "mutants/debug/"    
METHOD_LIST = ["remut"]
# METHOD_LIST = ["NOremut"]
# ATTENTION_METHOD = "mth5"
# ATTENTION_METHOD = "mth1"
#Mutations CONFIG

TF_KERAS_VIS_TECHNIQUE = "Gradcam++"
# TF_KERAS_VIS_TECHNIQUE = "Faster-ScoreCAM"
# ATTENTION_METHOD = "distances"
ATTENTION_METHOD = "probability"
SAVE_IMAGES = True
START_INDEX_DATASET = 0
EXTENT = 0.2
EXTENT_STEP = 0.1
EXTENT_LOWERBOUND = 0.2
EXTENT_UPPERBOUND = 2
NUMBER_OF_POINTS = 6
SQUARE_SIZE = 3
NUMBER_OF_MUTATIONS = 500
NUMBER_OF_REPETITIONS = 5
SEEDS_LIST_FOR_REPETITIONS_OF_MUTATIONS = [4398, 980, 987423, 99982, 1123, 4098, 1946, 22601, 55037, 812109, 53898, 187988]
NORMAL_MUTATION_ADAPTIVE_ENABLED = True

#Adaptive number of Mutations CONFIG
FITNESS_THRESHOLD_TO_GENERATE_MORE_MUTATIONS = 0.8 #The script will add more mutations if the Fitness is less than the THRESHOLD
EXTRA_MUTATIONS = 1 #Number of mutations to be added

#Init Data Set CONFIG
RANDOM_SEED = 2
SHUFFLE_IMAGES = False
NUMBER_OF_DIGIT_SAMPLES = 5 #Number of samples of a same digit that will be mutated in the experiment
N = 10 * NUMBER_OF_DIGIT_SAMPLES #Nummber of total Images to take from MNIST dataset

START_SEED = RANDOM_SEED
DEBUG_OR_VALID = "VALID"
# RUNNING_OPTION = "ATT_vs_NOR"
# RUNNING_OPTION = "ATT_vs_ATT+ADP_vs_NOR"
RUNNING_OPTION = "ATT_vs_ATT+ADP_vs_NOR_vs_NOR+ADP"
# RUNNING_OPTION = "VINCENZO_FUNCTIONS_DEMO"

#Specify the MNIST images (by the indexes) to reproduce results. If None, the list of images to be tested will be initialized randomly based on START_SEED.
# RUN_MNIST_SPECIFIC_INDEXES = [1439, 7750, 8880, 5518, 6546, 7616, 7979, 7329, 4445, 9439, 7403, 5509, 90, 7483, 4650, 4498, 8256, 9792, 5067, 2660]
RUN_MNIST_SPECIFIC_INDEXES = None #Specify the MNIST images (by the indexes) to reproduce results. If None, the list of images to be tested will be initialized randomly based on START_SEED.

SAVE_STATS4_CSV = False #Option to print the location of the points mutated at each iteration

#Histogram and Chart Generators Configs
HISTOGRAM1_PATH = "mutants/Histogram.png" #Plotting Plotting histogram comparison: four methods same image
HISTOGRAM2_PATH = "mutants/Histogram_1.png" #Plotting histogram comparison: Adaptive vs Without Adaptive (Top and Bottom)
HISTOGRAM3_PATH = "mutants/Histogram_2.png" #Plotting histogram comparison: Att vs Normal (Top and Bottom)