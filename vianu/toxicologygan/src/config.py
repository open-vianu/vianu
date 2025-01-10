from pathlib import Path

ROOT_PATH = Path(__file__).parents[2]

working_dir = ROOT_PATH / r"toxicologygan"

#######################################################################
# DEFAULT VALUES
#######################################################################

## Data_path : where data is stored
data_path: str = "./dummy_data/Example_Data_training.tsv"
## descriptors_path : path to molecular descriptor file
descriptors_path: str = "./dummy_data/Example_MolecularDescriptors.tsv"
## N_epochs : number of epochs of training
n_epochs: int = 10
## batch_size : size of batch
batch_size: int = 128
## lr : adam - learning rate
lr: float = 1e-7
## b1 : adam - decay of the first order momentum of the gradient
b1: float = 0.8
## b2 : adam - decay of second order momentum of the gradient
b2: float = 0.95
## n_cpu : number of cpu threads to use during batch generation
n_cpu: int = 8
## latent_dim / Zdim : dimension of the latent space (noise)
latent_dim: int = 1828
## Stru_dim : dimension of molecular descriptors
molecular_dim: int = 1826
## Time_dim : dimension of sacrificed time point (4,8,15,29 days)
Time_dim: int = 1
## Dose_dim : dimension of dose level (low:middle:high=1:3:10)
Dose_dim: int = 1
## Measurement_dim : dimension of Hematology and Biochemistry measurements
Measurement_dim: int = 38
## n_critic : number of critic iterations per generator iteration
n_critic: int = 5
## interval : number of intervals you want to save models
interval: int = 5
## lambda_gp : strength of the gradient penalty regularization term
lambda_gp: float = 1.0
## lambda_GR : strength of the regularization term for generator
lambda_GR: float = 0.2
## model_path : path to model saving folder
model_path: str = "./models"
## filename_Losses : filename of losses
filename_Losses: str = "Loss.txt"
## num_generate : number of blood testing records you want to generate
num_generate: int = 100
