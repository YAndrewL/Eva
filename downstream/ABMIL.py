import torch
import torch.nn as nn

class GatedABMIL(nn.Module):
    """
    See Attention Based MIL: https://arxiv.org/pdf/1802.04712,
    and implementation from: https://github.com/bunnelab/virtues/blob/main/models/ABMIL/gated_abmil.py
    """

    def __init__(
        self,
        emb_dim=None,
        hidden_dim=None,
        num_heads=1,
        feature_extractor=None,
        classifier=None,
        learnable_values=False,
    ) -> None:
        super().__init__()

        self.V = nn.Linear(emb_dim, hidden_dim)
        self.U = nn.Linear(emb_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, num_heads)

        if learnable_values:
            assert emb_dim is not None and hidden_dim is not None
            self.value_proj = nn.Linear(emb_dim, emb_dim)
        else:
            self.value_proj = nn.Identity()

        self.num_heads = num_heads

        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = nn.Identity()

        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = nn.Identity()

    def forward(self, x, mask=None):
        """
        x: input of size B x S x D
        mask: mask of size B x S indicating padding (0)
        """
        x = self.feature_extractor(x)

        v_x = self.V(x)
        u_x = self.U(x)

        v_x = nn.functional.tanh(v_x)
        u_x = nn.functional.sigmoid(u_x)

        h = v_x * u_x

        attn_scores = self.W(h)  # B x S x H
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(2), -1e9)
        attn_weights = nn.functional.softmax(attn_scores, dim=1)  # B x S x H
        attn_weights = attn_weights.transpose(1, 2)  # B x H x S

        output = torch.bmm(attn_weights, self.value_proj(x))  # B x H x D
        output_flat = output.reshape(-1, self.num_heads * x.size(2))

        return output_flat

    def compute_attention(self, x, mask=None, batched=True):
        """
        x: input of size B x S x D
        """
        if not batched:
            x = x.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        if x.dim() > 3:
            reshaped = True
            old_shape = x.shape
            x = x.reshape(x.size(0), -1, x.size(-1))
        else:
            reshaped = False

        x = self.feature_extractor(x)
        v_x = self.V(x)
        u_x = self.U(x)

        v_x = nn.functional.tanh(v_x)
        u_x = nn.functional.sigmoid(u_x)

        h = v_x * u_x

        attn_scores = self.W(h)  # B x S x H
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)
        attn_weights = nn.functional.softmax(attn_scores, dim=1)  # B x S x H
        attn_weights = attn_weights.transpose(1, 2)  # B x H x S

        if reshaped:
            attn_weights = attn_weights.reshape(old_shape[0], self.num_heads, *old_shape[1:-1])
        if not batched:
            attn_weights = attn_weights.squeeze(0)

        return attn_weights
