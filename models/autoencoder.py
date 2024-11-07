import torch
import torch.nn as nn
from . import layers

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, encoder_spec=None, decoder_spec=None):
        super(Autoencoder, self).__init__()
        
        # Default encoder and decoder specifications
        if encoder_spec is None:
            encoder_spec = [128, 'interact', 64, latent_dim]
        if decoder_spec is None:
            decoder_spec = encoder_spec[::-1]  # Mirror encoder for decoder
        
        # Encoder and decoder using shared parse_specification
        self.encoder_layers = layers.layer_spec(encoder_spec, input_dim, final_dim=latent_dim)
        self.decoder_layers = layers.layer_spec(decoder_spec, latent_dim, final_dim=input_dim)

    def forward(self, x):
        # Encoder forward pass
        for layer in self.encoder_layers:
            if isinstance(layer, layers.InteractionLayer):
                x = torch.cat((x, layer(x)), dim=1)
            else:
                x = layer(x)
        
        # Latent representation
        latent = x
        
        # Decoder forward pass
        for layer in self.decoder_layers:
            if isinstance(layer, layers.InteractionLayer):
                x = torch.cat((x, layer(x)), dim=1)
            else:
                x = layer(x)
        
        # Final reconstruction
        reconstructed = x
        
        return reconstructed, latent