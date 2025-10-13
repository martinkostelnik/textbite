import lightning as L
import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, hidden_dim if n_layers > 1 else output_dim)])

        for i in range(n_layers - 1):
            self.layers.extend([torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim if i < n_layers - 2 else output_dim)])

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, 4]
        """
        for layer in self.layers:
            x = layer(x)

        return x

class TextBiteMLPEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        text_embedder_input_dim: int,
        n_layers: int,
    ):
        super().__init__()

        self.position_embedder = MLP(
            input_dim=4,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_layers=n_layers,
        )

        self.text_embedder = MLP(
            input_dim=text_embedder_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_layers=n_layers,
        )

        self.encoder = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_layers=n_layers,
        )

    def forward(
        self,
        bboxes: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            bboxes: Tensor, shape [batch_size, seq_len, 4]
            text_embeddings: Tensor, shape [batch_size, seq_len, text_embedder_input_dim]
        """
        pos_embeds = self.position_embedder(bboxes)
        text_embeds = self.text_embedder(text_embeddings)

        combined = pos_embeds + text_embeds

        encoded = self.encoder(combined)
        return encoded
