"""Train the geodesic model: continuous flow from bulk to boundary.

Teacher: Qwen3-4B (discrete, 36 layers)
Student: Neural ODE — learns f(h, t) the bulk gradient field
Training: student output at t=1 must match teacher output

The student learns to flow from embedding (bulk) to resolved state
(boundary) in one continuous motion. No layers. No steps.

f(h, t) is a small network that outputs the direction to move.
torchdiffeq integrates it. The solver handles stability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import time
import os

device = "cuda"

print("=" * 70)
print("GEODESIC MODEL — continuous bulk-to-boundary flow")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in teacher.parameters():
    p.requires_grad = False

H = teacher.config.hidden_size  # 2560
VOCAB = teacher.config.vocab_size

print(f"Teacher loaded. H={H}, VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


# ═══════════════════════════════════════════════════════
# The bulk dynamics: f(h, t) = direction to move in the bulk
#
# This is a small network. It takes the current state h and
# the "time" t (position along the geodesic from 0=embedding
# to 1=boundary), and outputs dh/dt (how to move).
#
# Architecture: residual MLP with time conditioning.
# The MLP provides the "bulk medium" (holographic projection).
# Time conditioning provides the curve shape (rotation schedule).
# ═══════════════════════════════════════════════════════

class BulkDynamics(nn.Module):
    """f(t, h) — the vector field in the bulk.

    Small: just enough to define the flow direction at each point.
    Time t conditions the dynamics (early = fast rotation, late = slow).
    """
    def __init__(self, hidden_dim, intermediate=None):
        super().__init__()
        if intermediate is None:
            intermediate = hidden_dim * 2

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, hidden_dim),
        )

        # The dynamics: two-layer MLP with residual
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden_dim)

        # Initialize fc2 to small values so initial dynamics are gentle
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, t, h):
        """
        t: scalar (integration time, 0 to 1)
        h: [B*T, H] flattened hidden states
        Returns: dh/dt [B*T, H]
        """
        # Cast to float32 for dynamics computation
        h = h.float()

        # Time conditioning
        t_emb = self.time_embed(t.float().reshape(1, 1).expand(h.shape[0], 1))

        # Dynamics
        h_cond = self.norm1(h + t_emb)
        dh = self.fc2(F.silu(self.fc1(h_cond)))

        return dh


class GeodesicModel(nn.Module):
    """Continuous flow: embedding → ODE → boundary → lm_head."""

    def __init__(self, teacher_model):
        super().__init__()
        # Frozen from teacher
        self.embed = teacher_model.model.embed_tokens
        self.lm_head = teacher_model.lm_head
        for p in self.embed.parameters(): p.requires_grad = False
        for p in self.lm_head.parameters(): p.requires_grad = False

        # The learned dynamics (trainable)
        self.dynamics = BulkDynamics(H, intermediate=H * 4)

        # Final norm before lm_head
        self.final_norm = nn.LayerNorm(H)

    def forward(self, input_ids, integration_time=1.0):
        """
        input_ids: [B, T]
        Returns: logits [B, T, V]
        """
        h = self.embed(input_ids)  # [B, T, H]
        B, T, D = h.shape

        # Flatten for ODE (torchdiffeq expects [batch, features])
        h_flat = h.reshape(B * T, D)

        # Integrate from t=0 (bulk/embedding) to t=1 (boundary/resolved)
        t_span = torch.tensor([0.0, integration_time], device=device)

        # ODE solve
        h_trajectory = odeint(
            self.dynamics,
            h_flat,
            t_span,
            method='euler',  # simple euler for speed; can upgrade to dopri5
            options={'step_size': 0.05}  # ~20 internal steps
        )
        # h_trajectory: [2, B*T, H] — states at t=0 and t=1
        h_final = h_trajectory[-1]  # [B*T, H] — state at boundary

        # Reshape and project to vocab
        h_out = h_final.reshape(B, T, D)
        h_normed = self.final_norm(h_out.float())
        return self.lm_head(h_normed.to(self.lm_head.weight.dtype))


# ═══════════════════════════════════════════════════════
# Build and train
# ═══════════════════════════════════════════════════════

print("Building geodesic model...", flush=True)
student = GeodesicModel(teacher).to(device)

# Convert dynamics to float32 for stability, keep rest in bf16
student.dynamics = student.dynamics.float()
student.final_norm = student.final_norm.float()

trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
total = sum(p.numel() for p in student.parameters())
print(f"Params: {total/1e6:.0f}M total, {trainable/1e6:.1f}M trainable (dynamics + norm)")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# Training data
texts = [
    "The history of mathematics spans thousands of years and includes contributions from many civilizations.",
    "Marine biology studies organisms in the ocean covering more than seventy percent of Earth surface.",
    "Artificial intelligence has progressed through several distinct phases since its inception in the 1950s.",
    "Quantum mechanics describes matter and energy at smallest scales where particles exhibit wave properties.",
    "The French Revolution transformed French society by uprooting centuries of tradition and absolute monarchy.",
    "Climate change driven by burning fossil fuels threatens ecosystems worldwide through rising temperatures.",
    "The human genome contains approximately three billion base pairs organized into twenty three chromosomes.",
    "Machine learning algorithms improve through experience without being explicitly programmed for each task.",
    "The Amazon rainforest produces significant oxygen and houses incredible biodiversity across many species.",
    "Cryptography enables secure communication by transforming readable messages into encrypted ciphertext.",
    "Neural networks learn hierarchical representations through multiple layers of nonlinear transformations.",
    "General relativity describes gravity as curvature of spacetime caused by mass and energy distributions.",
    "Evolution explains how populations change over generations through variation inheritance and selection.",
    "The periodic table organizes chemical elements by atomic number revealing patterns in their properties.",
    "Photosynthesis converts light energy into chemical energy using water and carbon dioxide in plants.",
    "The Internet connected billions of devices through standardized protocols enabling communication worldwide.",
] * 6

train_ids = []
for text in texts:
    toks = tokenizer(text, return_tensors='pt', truncation=True,
                     max_length=48, padding='max_length').input_ids[0]
    train_ids.append(toks)
train_ids = torch.stack(train_ids).to(device)
print(f"Training data: {train_ids.shape}")

# Optimizer — very small LR to prevent NaN from large initial loss
optimizer = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad],
    lr=1e-5, weight_decay=0.01
)

N_STEPS = 500
BATCH = 4

print(f"\nTraining {N_STEPS} steps, batch={BATCH}")
print(f"Phase 1 (steps 0-200): MSE on hidden states (stable warmup)")
print(f"Phase 2 (steps 200-500): KL on logits (final objective)")
print(f"{'Step':>6} {'Loss':>10} {'Type':>5} {'VRAM':>6}")
print("-" * 35)

losses = []
t_start = time.time()

for step in range(N_STEPS):
    idx = torch.randint(0, len(train_ids), (BATCH,))
    batch = train_ids[idx]

    with torch.no_grad():
        t_out = teacher.model(batch, output_hidden_states=True)
        t_hidden_final = t_out.hidden_states[-1]  # [B, T, H]
        t_logits = teacher.lm_head(teacher.model.norm(t_hidden_final))

    # Student forward (through ODE)
    h_embed = student.embed(batch)
    B, T, D = h_embed.shape
    h_flat = h_embed.reshape(B * T, D)
    t_span = torch.tensor([0.0, 1.0], device=device)
    h_traj = odeint(student.dynamics, h_flat, t_span, method='euler',
                    options={'step_size': 0.05})
    h_final = h_traj[-1].reshape(B, T, D)

    if step < 200:
        # Phase 1: MSE on hidden states (match teacher's final hidden state)
        loss = F.mse_loss(h_final.float(), t_hidden_final.float())
        loss_type = "MSE"
    else:
        # Phase 2: KL on logits
        h_normed = student.final_norm(h_final.float())
        s_logits = student.lm_head(h_normed.to(student.lm_head.weight.dtype))
        TEMP = 2.0
        t_probs = F.softmax(t_logits.float() / TEMP, dim=-1)
        s_log_probs = F.log_softmax(s_logits.float() / TEMP, dim=-1)
        loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (TEMP ** 2)
        loss_type = "KL"

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
    optimizer.step()

    losses.append(loss.item())

    if step % 25 == 0 or step == N_STEPS - 1:
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"{step:>6} {loss.item():>10.4f} {loss_type:>5} {vram:>5.1f}G", flush=True)

elapsed = time.time() - t_start
print(f"\nTraining: {elapsed:.0f}s ({N_STEPS/elapsed:.1f} steps/s)")

# ═══════════════════════════════════════════════════════
# Validation: generate and compare
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("VALIDATION")
print(f"{'='*60}")

for prompt in ["The future of artificial intelligence will",
               "Water freezes at zero degrees and boils at",
               "The most important discovery in physics was"]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    # Teacher
    with torch.no_grad():
        t_out = teacher.generate(ids, max_new_tokens=30, do_sample=False)
    t_text = tokenizer.decode(t_out[0][ids.shape[1]:], skip_special_tokens=True)

    # Student (autoregressive with ODE)
    with torch.no_grad():
        gen = ids.clone()
        for _ in range(30):
            logits = student(gen)
            next_tok = logits[0, -1:].argmax(-1)
            gen = torch.cat([gen, next_tok.unsqueeze(0)], dim=-1)
    s_text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)

    print(f"\n  '{prompt}'")
    print(f"  Teacher: {t_text[:70]}")
    print(f"  Student: {s_text[:70]}")

# Save
SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save({
    "dynamics_state": student.dynamics.state_dict(),
    "norm_state": student.final_norm.state_dict(),
    "config": {"hidden": H, "intermediate": H * 4, "method": "euler",
               "step_size": 0.05, "integration_time": 1.0},
    "losses": losses,
}, os.path.join(SAVE_DIR, "geodesic_4b.pt"))
print(f"\nSaved. Final KL: {losses[-1]:.4f}", flush=True)
