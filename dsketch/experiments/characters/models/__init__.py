import torchbearer

from dsketch.experiments.shared.utils import list_class_names
from .encoders import *
from .model_bases import get_model
from .recurrent_decoders import *
from .single_pass_decoders import *

MU, LOGVAR = torchbearer.state_key('mu'), torchbearer.state_key('logvar')


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def get_feature(self, x):
        return self.encoder(x, None)

    def forward(self, x, state=None):
        x = self.encoder(x, state)

        return self.decoder(x, state)

    def get_callbacks(self, args):
        return [*self.encoder.get_callbacks(args), *self.decoder.get_callbacks(args)]


class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, latent_size):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size  # note enc will be made to emit 2x latent size output & we'll split

    # Sampling function (using the reparameterisation trick)
    def _sample(self, mu, log_sigma2):
        if self.training:
            eps = torch.randn(mu.shape, device=mu.device)
            return mu + torch.exp(log_sigma2 / 2) * eps
        else:
            return mu

    def get_feature(self, x):
        return self.encoder(x, None)[:, 0:self.latent_size]

    def forward(self, x, state=None):
        x = self.encoder(x)

        state[MU] = x[:, 0:self.latent_size]
        state[LOGVAR] = mu = x[:, self.latent_size:]

        z = self._sample(state[MU], state[LOGVAR])
        images = self.decoder(z, state)

        return images

    def get_callbacks(self, args):
        return [*self.encoder.get_callbacks(args), *self.decoder.get_callbacks(args)]


def model_choices(clz):
    return list_class_names(clz, __package__)
