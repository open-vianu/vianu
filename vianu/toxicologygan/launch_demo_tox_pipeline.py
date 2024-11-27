import os
import torch
#from utils import Time, Dose
import pandas as pd

from src.client import ToxicologyGANClient, GeneratorModel, Discriminator
from src.config import working_dir
from src.config import data_path, descriptors_path, batch_size, latent_dim, molecular_dim, Time_dim, Dose_dim, Measurement_dim
from src.config import n_epochs, n_critic, lr, b1, b2, interval, model_path, lambda_gp, lambda_GR, num_generate

# Change PATH to working directory - for example: working_dir = Path(__file__).parents[2] / r'toxicologygan'
os.chdir(working_dir)

# Device config for torch routines
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the client
tg_client = ToxicologyGANClient()

# Consume data into client
dataloader = tg_client.create_custom_dataloader(data_path, descriptors_path, batch_size, device)

# Instantiate Generator and Discriminator
generator = GeneratorModel(latent_dim, molecular_dim, Time_dim, Dose_dim, Measurement_dim).to(device)
discriminator = Discriminator(molecular_dim, Time_dim, Dose_dim, Measurement_dim).to(device)

# Training WGAN-GP with generator regularization
tg_client.train(generator, discriminator, dataloader, n_epochs, n_critic, latent_dim, device, lr, b1, b2, interval,
                model_path, lambda_gp, lambda_GR)


# USE TRAINED MODEL TO GENERATE NEW DATA
path = working_dir

# Read data
treatments = pd.read_csv(os.path.join(path, 'dummy_data', 'Example_Treatments_test.tsv'), sep="\t")
training_data = pd.read_csv(os.path.join(path, 'dummy_data', 'Example_Data_training.tsv'), sep="\t")  ### this file should store all the training data used for the pretrained model
MDs = pd.read_csv(os.path.join(path, 'dummy_data', 'Example_MolecularDescriptors.tsv'), index_col=0, sep="\t")
descriptors_training = MDs[MDs.index.isin(training_data['COMPOUND_NAME'])]
descriptors = MDs[MDs.index.isin(treatments['COMPOUND_NAME'])]

# Instantiate new generator model
generator = GeneratorModel(latent_dim, molecular_dim, Time_dim, Dose_dim, Measurement_dim).to(device)

# Load model
model_path = os.path.join(path, 'models', 'generator_10')
weights = torch.load(str(model_path))
generator.load_state_dict(weights)
generator.eval()

# Generate and save
result_path = os.path.join(path, 'results', 'generated_data_{}.tsv'.format(num_generate))
tg_client.generate(treatments, descriptors, training_data, descriptors_training, result_path, generator, device, num_generate, latent_dim)