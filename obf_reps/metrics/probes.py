import torch
import torch.nn.functional as F


class LogisticRegression(torch.nn.Module):
    """Differentiable Logistic Regression.

    We could use LogisicRegression from sklearn, but it is not differentiable.
    """

    def __init__(self, input_dim, dtype=torch.float16):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1, dtype=dtype)

    def forward(self, x):
        return self.linear(x)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dtype=torch.float16):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1, dtype=dtype),
        )

    def forward(self, x):
        return self.model(x)


class AttentionProbe(torch.nn.Module):
    """
    Multi-head attention-style probe (sequence-level), without positional bias and
    without factorizing the value projection. By default, expands the single
    sequence logit to token-level logits so it fits the existing TrainableMetric
    interface (i.e., returns [B, S]).

    Given A ∈ R^{S×D} (or [B, S, D]):
      Q = A W_q  ∈ R^{S×H}
      V = A W_v  ∈ R^{S×H}
      α = softmax(Q, dim=seq)  over the sequence dimension
      z_h = Σ_s α[s,h] * V[s,h]   (per-head summary)
      logit = z W_o + b  ∈ R^{1}

    Args:
        d_model: hidden size D
        n_heads: number of heads H
        tokenwise_output: if True (default), repeat the sequence logit across
            the sequence dimension to produce [B, S]; if False, return [B, 1].
        dtype: dtype for parameters

    Inputs:
        hidden_states: [S, D] or [B, S, D]
        attention_mask (optional): bool mask [S] or [B, S]; True = keep, False = mask

    Returns:
        If tokenwise_output=True:  [S] or [B, S]  (logits per token, same value)
            - bit of a hack to be compatible with TrainableMetric
        If tokenwise_output=False: [1] or [B, 1]  (single logit per sequence)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        tokenwise_output: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.tokenwise_output = tokenwise_output

        # Per-token scalar queries/values per head (no factorization)
        self.q_proj = torch.nn.Linear(d_model, n_heads, bias=False, dtype=dtype)
        self.v_proj = torch.nn.Linear(d_model, n_heads, bias=False, dtype=dtype)

        # Combine head summaries into a single logit
        self.out_proj = torch.nn.Linear(n_heads, 1, bias=True, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = hidden_states
        orig_dim = x.dim()

        if orig_dim == 2:
            # [S, D] -> [1, S, D]
            x = x.unsqueeze(0)
        elif orig_dim == 3:
            # [B, S, D]
            pass
        else:
            raise ValueError(f"Expected [S, D] or [B, S, D], got {tuple(x.shape)}")

        B, S, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {D}")

        # Projections: [B, S, H]
        q = self.q_proj(x)
        v = self.v_proj(x)

        # Attention logits over sequence (per head); do softmax in float32
        attn_logits = q.to(torch.float32)

        if attention_mask is not None:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)  # [1, S]
            if attention_mask.shape != (B, S):
                raise ValueError(
                    f"attention_mask must have shape [B, S]; got {tuple(attention_mask.shape)}"
                )
            attn_logits = attn_logits.masked_fill(
                ~attention_mask[:, :, None].to(torch.bool), float("-inf")
            )

        attn = F.softmax(attn_logits, dim=1).to(q.dtype)  # [B, S, H]

        # Weighted sum across sequence -> [B, H]
        head_summaries = (attn * v).sum(dim=1)

        # Single sequence logit -> [B, 1]
        logits_seq = self.out_proj(head_summaries)  # [B, 1]

        if self.tokenwise_output:
            # Expand the single logit across tokens
            if orig_dim == 2:
                # Return [S]
                logits = logits_seq.squeeze(0).squeeze(-1).expand(S)
            else:
                # Return [B, S]
                logits = logits_seq.squeeze(-1).unsqueeze(-1).expand(B, S).contiguous()
        else:
            # Return one logit per sequence
            if orig_dim == 2:
                logits = logits_seq.squeeze(0)  # [1]
            else:
                logits = logits_seq  # [B, 1]

        return logits
