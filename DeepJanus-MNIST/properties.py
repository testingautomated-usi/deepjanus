# Make sure that any of this properties can be overridden using env.properties
import os

# GA Setup
POPSIZE          = int(os.getenv('DJ_POPSIZE', '800'))
NGEN             = int(os.getenv('DJ_NGEN', '500000'))

RUNTIME          = int(os.getenv('DJ_RUNTIME', '3600'))
INTERVAL         = int(os.getenv('DJ_INTERVAL', '900'))

# Mutation Hyperparameters
# range of the mutation
MUTLOWERBOUND    = float(os.getenv('DJ_MUTLOWERBOUND', '0.01'))
MUTUPPERBOUND    = float(os.getenv('DJ_MUTUPPERBOUND', '0.6'))

# Dataset
EXPECTED_LABEL   = int(os.getenv('DJ_EXPECTED_LABEL', '5'))

#------- NOT TUNING ----------

# mutation operator probability
MUTPB            = float(os.getenv('DJ_MUTPB', '0.7'))
MUTOPPROB        = float(os.getenv('DJ_MUTOPPROB', '0.5'))
MUTOFPROB        = float(os.getenv('DJ_MUTOFPROB', '0.5'))


IMG_SIZE         = int(os.getenv('DJ_IMG_SIZE', '28'))
num_classes      = int(os.getenv('DJ_NUM_CLASSES', '10'))

INITIALPOP       = os.getenv('DJ_INITIALPOP', 'seeded')

MODEL            = os.getenv('DJ_MODEL', 'models/model_mnist.h5')

ORIGINAL_SEEDS   = os.getenv('DJ_ORIGINAL_SEEDS', 'bootstraps_five')

BITMAP_THRESHOLD = float(os.getenv('DJ_BITMAP_THRESHOLD', '0.5'))

DISTANCE         = float(os.getenv('DJ_DISTANCE', '2.0'))


# Reseeding Hyperparameters
# extent of the reseeding operator
RESEEDUPPERBOUND = int(os.getenv('DJ_RESEEDUPPERBOUND', '10'))


K_SD             = float(os.getenv('DJ_K_SD', '0.1'))

# K-nearest
K                = int(os.getenv('DJ_K', '1'))

# Archive configuration
ARCHIVE_THRESHOLD = float(os.getenv('ARCHIVE_THRESHOLD', '2.0'))

GENERATE_ONE_ONLY = False

# Directories
PATH              = os.getenv('PATH','vectorized_images_GA')
TESTSOURCEPATH    = os.getenv('TESTSOURCEPATH','source_images_GA')
TRAINSOURCEPATH   = os.getenv('TRAINSOURCEPATH', 'source_images_trainset')
RESULTS_PATH      = os.getenv('RESULTS_PATH', 'results')