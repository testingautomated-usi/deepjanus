import os

# GA Setup
POPSIZE = 12
NGEN = 100

MISB_TSHD = 5.0

# Reseeding Hyperparameters
# extent of the reseeding operator
RESEEDUPPERBOUND = 2

K_SD = 0.01

# K-nearest
K = 1

# Archive configuration
ARCHIVE_THRESHOLD = 0.7


#------- NOT TUNING ----------

# mutation operator probability
MUTOPPROB = 0.5
MUTOFPROB = 0.5

IMG_SIZE = 28
num_classes = 10


INITIALPOP = 'seeded'

GENERATE_ONE_ONLY = False

MODEL = 'models/lenet_original_13.h5'


RESULTS_PATH = 'results'

DATASET = "populations//population_9"
SIKULIX_API_PATH = "Sikuli-jars//sikulixapi-2.0.4.jar"

SIKULIX_REST = "http://localhost:50001/"

SIKULIX_SCRIPT_FOLDER = os.path.join("C://","sikulix_scripts")

#SIKULIX_SCRIPT_PATH = 'C:\\sikuliscripts\\unityeyes.sikuli'
SIKULIX_SCRIPT_PATH = 'C://Users//vinni//PycharmProjects//DeepMetis-UnityEyes//sikulix_scripts//unityeyes.sikuli'


SIKULIX_SCRIPTS_HOME = "sikulix_scripts"

SIKULIX_SCRIPT_NAME = "unityeyes"

SIKULIX_SCRIPT_NAME_W_FMT = "unityeyes.sikuli"

SIKULIX_ANGLES_FILE_NAME = "angles.txt"

# TODO: modify unityeyes paths and interpreter
UNITYEYES_PATH = "C://Users//vinni//Desktop//UnityEyes//UnityEyes_Windows"
INTERPRETER = r'C:\Users\vinni\PycharmProjects\venvs\DeepMetis-UnityEyes\Scripts\python'

UNITY_GENERATED_IMGS_PATH = UNITYEYES_PATH + "//imgs_"
UNITY_STANDARD_IMGS_PATH = UNITYEYES_PATH + "//imgs//"
