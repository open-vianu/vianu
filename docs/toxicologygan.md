---
title: ToxicologyGAN
layout: home
nav_order: 3
---

# Welcome to ToxicologyGAN

**ToxicologyGAN** is an out-of-the-box tool within the vianu package to help users facilitate the training of a GAN
network in the context of simulating ...

[Skip to Installation](#installation-guide){: .btn .btn-purple }

## Features

1. **Training of GAN Network**: Start by loading the relevant data and training a Generator and a Discriminator network.

2. **Generation**: Proceed with an already trained network and use the Generator with the corresponding trained weights
to generate new data points.

## How it works

Based on the work of [Chen et al. (Nature Communications, 2023)](https://www.nature.com/articles/s41467-023-42933-9), 
we implemented a simpler version of the code to make it more accessible and easy to use. Once having the correct input 
data, the training of the network is easily performed with the implementation in vianu.


### Prerequisites

- Python 3.11 or higher
- Required Python packages

## Installation guide

1. **Download package**

   ```bash
   pip install vianu
   ```
   
2. **Use Pipeline to train GAN network and generate new data points**: In Python you can use the following:
    ```python
        from src.client import ToxicologyGANClient, GeneratorModel, Discriminator
        from src.config import working_dir
        from src.config import (data_path, descriptors_path, batch_size, latent_dim, molecular_dim, Time_dim,
        Dose_dim,Measurement_dim)
        from src.config import n_epochs, n_critic, lr, b1, b2, interval, model_path, lambda_gp, lambda_GR, num_generate
        
        # Change PATH to working directory - for example: working_dir = Path(__file__).parents[2] / r'toxicologygan'
        os.chdir(working_dir)
        
        # Device config for torch routines
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate the client
        tg_client = ToxicologyGANClient()
        
        # Consume data into client
        dataloader = tg_client.create_custom_dataloader(data_path, descriptors_path, batch_size, device)
        
        # Instantiate Generator and Discriminator
        generator = GeneratorModel(latent_dim, molecular_dim, Time_dim, Dose_dim, Measurement_dim).to(device)
        discriminator = Discriminator(molecular_dim, Time_dim, Dose_dim, Measurement_dim).to(device)
        
        # Training WGAN-GP with generator regularization
        tg_client.train(generator, discriminator, dataloader, n_epochs, n_critic, latent_dim, device, lr, b1, b2, 
                        interval, model_path, lambda_gp,lambda_GR)
        
        # USE TRAINED MODEL TO GENERATE NEW DATA
        path = working_dir
        
        # Read data
        treatments = pd.read_csv(os.path.join(path, "dummy_data", "Example_Treatments_test.tsv"), sep="\t")
        training_data = pd.read_csv(os.path.join(path, "dummy_data", "Example_Data_training.tsv"), sep="\t")  ### this file should store all the training data used for the pretrained model
        MDs = pd.read_csv(os.path.join(path, "dummy_data", "Example_MolecularDescriptors.tsv"),index_col=0, sep="\t")
        descriptors_training = MDs[MDs.index.isin(training_data["COMPOUND_NAME"])]
        descriptors = MDs[MDs.index.isin(treatments["COMPOUND_NAME"])]
        
        # Instantiate new generator model
        generator = GeneratorModel(latent_dim, molecular_dim, Time_dim, Dose_dim, Measurement_dim).to(device)
        
        # Load model
        model_path = os.path.join(path, "models", "generator_10")
        weights = torch.load(str(model_path))  # nosec
        generator.load_state_dict(weights)
        generator.eval()
        
        # Generate and save
        result_path = os.path.join(path, "results", "generated_data_{}.tsv".format(num_generate))
        tg_client.generate(treatments,descriptors,training_data,descriptors_training,result_path,generator,device,
                           num_generate,latent_dim,)
    ```
