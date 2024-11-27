import os
import torch
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import autograd
from typing import Union
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("toxicologygan_logger")


class ToxicologyGANClient:
    """
    The client that orchestrates the data fetching and training.
    """

    def __init__(self):
        """
        Initializes the ToxicologyGANClient

        Args:
            None
        """
        pass

    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples, Stru, Time, Dose, device):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1)).to(device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = discriminator(interpolates, Stru, Time, Dose)
        fake = torch.ones(real_samples.shape[0], 1).to(device)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def calc_generator_regularization(self, Stru, Time, Dose, z, device, generator):
        b_sz = Stru.shape[0]
        Stru1 = Stru
        Time1 = Time
        Dose1 = Dose
        idx=torch.randperm(b_sz)
        Stru2 = Stru[idx]
        Time2 = Time[idx]
        Dose2 = Dose[idx]

        # Sample random numbers epsilon
        epsilon = torch.rand(b_sz, 1, device=device)

        interpolated_Stru = epsilon*Stru1 + (1 - epsilon)*Stru2
        interpolated_Time = epsilon*Time1 + (1 - epsilon)*Time2
        interpolated_Dose = epsilon*Dose1 + (1 - epsilon)*Dose2

        #conditions1 = torch.cat([Stru1, Time1, Dose1], -1)
        #conditions2 = torch.cat([Stru2, Time2, Dose2], -1)
        #interpolated_conditions = epsilon * conditions1 + (1 - epsilon) * conditions2

        perturbation_std = 0.01
        # perturbations = torch.randn(b_sz, interpolated_conditions.shape[0])*perturbation_std
        # perturbated_conditions = interpolated_conditions + perturbations
        perturbated_Stru = interpolated_Stru + torch.randn(b_sz, interpolated_Stru.shape[1]).to(device)*perturbation_std
        perturbated_Time = interpolated_Time + torch.randn(b_sz, interpolated_Time.shape[1]).to(device)*perturbation_std
        perturbated_Dose = interpolated_Dose + torch.randn(b_sz, interpolated_Dose.shape[1]).to(device)*perturbation_std

        batch_interpolated_samples = generator(z.detach(),
                                               interpolated_Stru.detach(),
                                               interpolated_Time.detach(),
                                               interpolated_Dose.detach())

        batch_noise_samples = generator(z.detach(),
                                        perturbated_Stru.detach(),
                                        perturbated_Time.detach(),
                                        perturbated_Dose.detach())

        gp_loss = torch.nn.MSELoss()
        gp = gp_loss(batch_interpolated_samples, batch_noise_samples)
        return gp



    def train(self, generator, discriminator, dataloader, n_epochs, n_critic, Z_dim, device, lr, b1, b2, interval,
              model_path, lambda_gp, lambda_GR):
        '''
        Trains a Generative Adversarial Network (GAN) using a generator and discriminator.

        Args:
            generator: The generator model responsible for creating synthetic data.
            discriminator: The discriminator model that distinguishes real from synthetic data.
            dataloader (torch.utils.data.DataLoader): DataLoader providing the real data for training.
            n_epochs (int): Number of training epochs.
            n_critic (int): Number of discriminator updates per generator update.
            Z_dim (int): Dimensionality of the generator's input noise vector.
            device (torch.device): The device (CPU or GPU) to use for training.
            lr (float): Learning rate for both generator and discriminator optimizers.
            b1 (float): Beta1 hyperparameter for the Adam optimizer.
            b2 (float): Beta2 hyperparameter for the Adam optimizer.
            interval (int): Interval for logging training progress and saving model checkpoints.
            model_path (str): Path to save the trained model checkpoints.
            lambda_gp (float): Weight for the gradient penalty term in the loss function.
            lambda_GR (float): Weight for an optional loss term related to generator regularization (e.g., perceptual or other losses).

        Returns:
            None

        Notes:
            - This function implements the GAN training loop with an optional gradient penalty for stabilizing training.
            - The `lambda_gp` parameter controls the contribution of the gradient penalty to the loss.
            - The `lambda_GR` parameter allows incorporating additional generator-specific loss terms if needed.

        Example:
            train(generator, discriminator, dataloader, n_epochs=50, n_critic=5, Z_dim=100, device='cuda',
                  lr=0.0002, b1=0.5, b2=0.999, interval=100, model_path='./models',
                  lambda_gp=10, lambda_GR=0.1)
        '''
        ###################################################################################
        # First we start by defining optimizers for generator and discriminator
        ###################################################################################

        optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

        # Log initialization of training
        logger.info("STARTING TRAINING")

        # Training loop
        for epoch in range(n_epochs):
            for i, (Measurement, Stru, Time, Dose) in enumerate(dataloader):
                batch_size = Measurement.shape[0]

                #  Train Discriminator
                optimizer_D.zero_grad()
                # Sample noise
                z = torch.randn(batch_size, Z_dim).to(device)
                z = (z - z.min()) / (z.max() - z.min())
                z = 2 * z - 1
                gen_Measurement = generator(z, Stru, Time, Dose)
                validity_real = discriminator(Measurement, Stru, Time, Dose)
                validity_fake = discriminator(gen_Measurement.detach(), Stru, Time, Dose)

                # Compute the Wasserstein loss and gradient penalty for the discriminator
                gradient_penalty = self.compute_gradient_penalty(discriminator, Measurement, gen_Measurement, Stru, Time, Dose,
                                                            device)
                # Backpropagate and optimize the discriminator
                d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty

                d_loss.backward()
                optimizer_D.step()

                # Train the generator every n_critic steps
                if (epoch * len(dataloader) + i) % n_critic == 0:
                    gen_Measurement = generator(z, Stru, Time, Dose)
                    validity = discriminator(gen_Measurement, Stru, Time, Dose)

                    # Compute the Regularization term for genenerator LGR(G)
                    LGR = lambda_GR * self.calc_generator_regularization(Stru, Time, Dose, z, device, generator)
                    # Generator loss
                    g_loss = -torch.mean(validity) + LGR

                    # Update generator
                    optimizer_G.zero_grad()
                    g_loss.backward()
                    optimizer_G.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch + 1, n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item())
                )
            if (epoch + 1) % interval == 0:
                print(f'this is model path: {model_path}')
                if not os.path.exists(model_path):
                    print(model_path)
                    os.makedirs(model_path)
                torch.save(generator.state_dict(), os.path.join(model_path, 'generator_{}'.format(epoch + 1)))


    def create_custom_dataloader(self, data_filepath: str, descriptors_path: str, batch_size: int, device: Union[str, torch.device]):
        """
        Create a custom DataLoader for data and conditions.

        Args:
            data_filepath (str): Path to the data file.
            descriptors_path (str): Path to the molecular descriptors file.
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): True/False to indicate a re-shuffle during training of data being used.

        Returns:
            DataLoader: A DataLoader instance for your data and conditions.
        """
        # Read data from files
        data = pd.read_csv(data_filepath, sep="\t")
        descriptors = pd.read_csv(descriptors_path, index_col=0, sep="\t")

        # Preprocess data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        descriptors = pd.DataFrame(scaler.fit_transform(descriptors), columns=descriptors.columns, index=descriptors.index)
        data = data.iloc[:, 0:3].join(pd.DataFrame(scaler.fit_transform(data.iloc[:, 3:]), columns=data.columns[3:]))

        S = pd.DataFrame(columns=descriptors.columns)
        M = pd.DataFrame(columns=data.columns[3:])
        T = []
        D = []
        for i in range(len(data)):
            if data.iloc[i].COMPOUND_NAME in descriptors.index:
                S = pd.concat([S, descriptors[descriptors.index == data.iloc[i].COMPOUND_NAME]])
                subset_data = data.iloc[i, 3:].to_frame().T
                M = pd.concat([M, subset_data], ignore_index=True)
                T.append(self.Time(data.iloc[i].SACRI_PERIOD))
                D.append(self.Dose(data.iloc[i].DOSE_LEVEL))

        # Convert data to PyTorch tensors
        S = torch.tensor(S.to_numpy(dtype=np.float32), device=device)
        M = torch.tensor(M.to_numpy(dtype=np.float32), device=device)
        T = scaler.fit_transform(np.array(T, dtype=np.float32).reshape(len(T), -1))
        T = torch.tensor(T, device=device)
        D = scaler.fit_transform(np.array(D, dtype=np.float32).reshape(len(D), -1))
        D = torch.tensor(D, device=device)
        dataset = torch.utils.data.TensorDataset(M, S, T, D)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader

    def Time(self, SACRIFICE_PERIOD):
        switcher = {
            '4 day': 4 / 29,
            '8 day': 8 / 29,
            '15 day': 15 / 29,
            '29 day': 29 / 29
        }
        return switcher.get(SACRIFICE_PERIOD, 'error')

    def Dose(self, DOSE_LEVEL):
        switcher = {
            'Low': 0.1,
            'Middle': 0.3,
            'High': 1
        }
        return switcher.get(DOSE_LEVEL, 'error')

    def generate(self, treatments, descriptors, training_data, descriptors_training, result_path, generator, device, num_generate, latent_dim):
        '''
        Args:
        treatments (pd.DataFrame): treatment conditions of interest
        descriptors (pd.DataFrame): molecular descriptors of the compounds of interest
        training_data (pd.DataFrame): all the training data used for the pretrained model
        descriptors_training (pd.DataFrame): molecular descriptors of the compounds used to train the model
        result_path (str) : path to the file where you want to store the results
        '''


        scaler = MinMaxScaler(feature_range=(-1, 1))

        scaler.fit(descriptors_training)
        scaled_MDs = pd.DataFrame(scaler.transform(descriptors), columns=descriptors.columns, index=descriptors.index)
        S = pd.DataFrame()
        for i in range(len(treatments)):
            S = pd.concat([S, scaled_MDs[scaled_MDs.index == treatments.iloc[i].COMPOUND_NAME]])
        S = torch.tensor(S.to_numpy(dtype=np.float32), device=device)

        scaler.fit(training_data['SACRI_PERIOD'].apply(self.Time).to_numpy(dtype=np.float32).reshape(-1, 1))
        T = scaler.transform(treatments['SACRI_PERIOD'].apply(self.Time).to_numpy(dtype=np.float32).reshape(-1, 1))
        T = torch.tensor(T, device=device)

        scaler.fit(training_data['DOSE_LEVEL'].apply(self.Dose).to_numpy(dtype=np.float32).reshape(-1, 1))
        D = scaler.transform(treatments['DOSE_LEVEL'].apply(self.Dose).to_numpy(dtype=np.float32).reshape(-1, 1))
        D = torch.tensor(D, device=device)

        measurements = training_data.iloc[:, 3:]
        scaler.fit(measurements)
        Results = pd.DataFrame(columns=measurements.columns)
        for i in range(S.shape[0]):
            num = 0
            while num < num_generate:
                z = torch.randn(1, latent_dim).to(device)
                generated_records = generator(z, S[i].view(1, -1), T[i].view(1, -1), D[i].view(1, -1))
                generated_records = scaler.inverse_transform(generated_records.cpu().detach().numpy())
                check = np.sum(generated_records[:, 9:14])
                # Source code 95 < check < 105
                if 80 < check < 120:
                    num += 1
                    Results.loc[i] = generated_records.flatten()
        Results = pd.concat([treatments.loc[treatments.index.repeat(num_generate)].reset_index(drop=True), Results], axis=1)
        Results.to_csv(result_path, sep='\t', index=False)



class GeneratorModel(nn.Module):
    def __init__(self, Z_dim, Stru_dim, Time_dim, Dose_dim, Measurement_dim):
        super(GeneratorModel, self).__init__()

        def unitary_block(features_in, features_out, normalize=True):
            layers = [nn.Linear(features_in, features_out)]
            if normalize:
                layers.append(nn.BatchNorm1d(features_out, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            *unitary_block(Z_dim + Stru_dim + Time_dim + Dose_dim, 4096, normalize=False),
            *unitary_block(4096, 2048),
            *unitary_block(2048, 1024),
            *unitary_block(1024, 256),
            *unitary_block(256, 64),
            nn.Linear(64, Measurement_dim),
            nn.Tanh()
        )

    def forward(self, noise, Stru, Time, Dose):
        # Concatenate conditions and noise to produce input
        gen_input = torch.cat([noise, Stru, Time, Dose], -1)
        Measurement = self.model(gen_input)
        return Measurement


class Discriminator(nn.Module):
    def __init__(self, Stru_dim, Time_dim, Dose_dim, Measurement_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(Stru_dim + Time_dim + Dose_dim + Measurement_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, Measurement, Stru, Time, Dose):
        # Concatenate conditions and real_Measurement to produce input
        d_in = torch.cat((Measurement, Stru, Time, Dose), -1)
        validity = self.model(d_in)
        return validity