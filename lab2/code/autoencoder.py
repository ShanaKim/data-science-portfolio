import lightning as L
import torch
import torch.nn as nn

# Model1
# Baseline model
class Autoencoder(L.LightningModule):
    def __init__(
        self, optimizer_config=None, n_input_channels=8, patch_size=9, embedding_size=8
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config

        input_size = int(n_input_channels * (patch_size**2))
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, embedding_size),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_size),
            torch.nn.Unflatten(1, (n_input_channels, patch_size, patch_size)),
        )
    def forward(self, batch):
        """
        Forward pass through the network.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, n_input_channels, width, height)
        """

        # all the autencoder does is encode then decode the input tensor
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The training loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put is the MSE between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # you can consider other possible loss functions, or add additional terms
        # to this loss.
        # for instance, could it be good to add a term that encourages sparsity
        # in the embedding?

        # log the training loss for experiment tracking purposes
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The validation loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put for validation is the MSE
        # between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # log the validation loss for experiment tracking purposes
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # set up the optimizer.
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        """
        Embeds the input tensor.
        Args:
            x: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, embedding_size)
        """
        return self.encoder(x)


# Model2
# deeper layers
class Autoencoder_deeper(L.LightningModule):
    def __init__(
        self, optimizer_config=None, n_input_channels=8, patch_size=9, embedding_size=8
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config

        input_size = int(n_input_channels * (patch_size**2))
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(input_size, 256),  # Increased neurons
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, embedding_size),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, input_size),
            torch.nn.Unflatten(1, (n_input_channels, patch_size, patch_size)),
        )
    def forward(self, batch):
        """
        Forward pass through the network.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, n_input_channels, width, height)
        """

        # all the autencoder does is encode then decode the input tensor
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The training loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put is the MSE between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # you can consider other possible loss functions, or add additional terms
        # to this loss.
        # for instance, could it be good to add a term that encourages sparsity
        # in the embedding?

        # log the training loss for experiment tracking purposes
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The validation loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put for validation is the MSE
        # between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # log the validation loss for experiment tracking purposes
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # set up the optimizer.
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        """
        Embeds the input tensor.
        Args:
            x: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, embedding_size)
        """
        return self.encoder(x)

# Model3
# More Nodes per Layer
class Autoencoder_more_nodes(L.LightningModule):
    def __init__(
        self, optimizer_config=None, n_input_channels=8, patch_size=9, embedding_size=8
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config

        input_size = int(n_input_channels * (patch_size**2))
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(input_size, 512),  # More nodes
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, embedding_size),)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, input_size),
            torch.nn.Unflatten(1, (n_input_channels, patch_size, patch_size)),)
    def forward(self, batch):
        """
        Forward pass through the network.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, n_input_channels, width, height)
        """

        # all the autencoder does is encode then decode the input tensor
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The training loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put is the MSE between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # you can consider other possible loss functions, or add additional terms
        # to this loss.
        # for instance, could it be good to add a term that encourages sparsity
        # in the embedding?

        # log the training loss for experiment tracking purposes
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The validation loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put for validation is the MSE
        # between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # log the validation loss for experiment tracking purposes
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # set up the optimizer.
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        """
        Embeds the input tensor.
        Args:
            x: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, embedding_size)
        """
        return self.encoder(x)


# Model4
# use LeakyReLU instead of ReLU
class Autoencoder_leakyReLU(L.LightningModule):
    def __init__(
        self, optimizer_config=None, n_input_channels=8, patch_size=9, embedding_size=8
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config

        input_size = int(n_input_channels * (patch_size**2))
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(input_size, 128),
            torch.nn.LeakyReLU(0.1),  
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(64, embedding_size),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 64),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(128, input_size),
            torch.nn.Unflatten(1, (n_input_channels, patch_size, patch_size)),
        )
        
    def forward(self, batch):
        """
        Forward pass through the network.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, n_input_channels, width, height)
        """

        # all the autencoder does is encode then decode the input tensor
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The training loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put is the MSE between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # you can consider other possible loss functions, or add additional terms
        # to this loss.
        # for instance, could it be good to add a term that encourages sparsity
        # in the embedding?

        # log the training loss for experiment tracking purposes
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The validation loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put for validation is the MSE
        # between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # log the validation loss for experiment tracking purposes
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # set up the optimizer.
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        """
        Embeds the input tensor.
        Args:
            x: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, embedding_size)
        """
        return self.encoder(x)


# Model5
# Use ELU 
class Autoencoder_ELU(L.LightningModule):
    def __init__(
        self, optimizer_config=None, n_input_channels=8, patch_size=9, embedding_size=8
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config

        input_size = int(n_input_channels * (patch_size**2))
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(input_size, 128),
            torch.nn.ELU(alpha=1.0),  
            torch.nn.Linear(128, 64),
            torch.nn.ELU(alpha=1.0),
            torch.nn.Linear(64, embedding_size),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 64),
            torch.nn.ELU(alpha=1.0),
            torch.nn.Linear(64, 128),
            torch.nn.ELU(alpha=1.0),
            torch.nn.Linear(128, input_size),
            torch.nn.Unflatten(1, (n_input_channels, patch_size, patch_size)),
        )


        
    def forward(self, batch):
        """
        Forward pass through the network.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, n_input_channels, width, height)
        """

        # all the autencoder does is encode then decode the input tensor
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The training loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put is the MSE between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # you can consider other possible loss functions, or add additional terms
        # to this loss.
        # for instance, could it be good to add a term that encourages sparsity
        # in the embedding?

        # log the training loss for experiment tracking purposes
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The validation loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put for validation is the MSE
        # between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # log the validation loss for experiment tracking purposes
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # set up the optimizer.
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        """
        Embeds the input tensor.
        Args:
            x: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, embedding_size)
        """
        return self.encoder(x)



# Model6
# Use SELU
class Autoencoder_SELU(L.LightningModule):
    def __init__(
        self, optimizer_config=None, n_input_channels=8, patch_size=9, embedding_size=8
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config

        input_size = int(n_input_channels * (patch_size**2))
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(input_size, 128),
            torch.nn.SELU(),  
            torch.nn.Linear(128, 64),
            torch.nn.SELU(),
            torch.nn.Linear(64, embedding_size),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 64),
            torch.nn.SELU(),
            torch.nn.Linear(64, 128),
            torch.nn.SELU(),
            torch.nn.Linear(128, input_size),
            torch.nn.Unflatten(1, (n_input_channels, patch_size, patch_size)),
        )


        
    def forward(self, batch):
        """
        Forward pass through the network.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, n_input_channels, width, height)
        """

        # all the autencoder does is encode then decode the input tensor
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The training loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put is the MSE between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # you can consider other possible loss functions, or add additional terms
        # to this loss.
        # for instance, could it be good to add a term that encourages sparsity
        # in the embedding?

        # log the training loss for experiment tracking purposes
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The validation loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put for validation is the MSE
        # between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # log the validation loss for experiment tracking purposes
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # set up the optimizer.
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        """
        Embeds the input tensor.
        Args:
            x: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, embedding_size)
        """
        return self.encoder(x)


# Model7
# Combine deeper layer and more nodes
class Autoencoder_combined(L.LightningModule):
    def __init__(
        self, optimizer_config=None, n_input_channels=8, patch_size=9, embedding_size=8
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config

        input_size = int(n_input_channels * (patch_size**2))
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(input_size, 512),  # More nodes
            torch.nn.ReLU(),  
            torch.nn.Linear(512, 256),  
            torch.nn.ReLU(),  
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),  
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),  
            torch.nn.Linear(64, embedding_size),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, input_size),
            torch.nn.Unflatten(1, (n_input_channels, patch_size, patch_size)),
        )


        
    def forward(self, batch):
        """
        Forward pass through the network.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, n_input_channels, width, height)
        """

        # all the autencoder does is encode then decode the input tensor
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The training loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put is the MSE between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # you can consider other possible loss functions, or add additional terms
        # to this loss.
        # for instance, could it be good to add a term that encourages sparsity
        # in the embedding?

        # log the training loss for experiment tracking purposes
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The validation loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put for validation is the MSE
        # between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # log the validation loss for experiment tracking purposes
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # set up the optimizer.
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        """
        Embeds the input tensor.
        Args:
            x: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, embedding_size)
        """
        return self.encoder(x)