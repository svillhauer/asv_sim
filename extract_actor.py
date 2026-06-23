"""
Run on HOST (numpy>=2.0, stable-baselines3 installed):
    python3 extract_actor.py

Extracts the SAC actor MLP weights as plain numpy arrays so the container
can run deterministic inference without SB3 or a matching numpy version.
"""
import numpy as np
import torch
from stable_baselines3 import SAC
from pathlib import Path

MODEL = Path(__file__).parent / "reference/trained_policy/best_model.zip"
OUT   = Path(__file__).parent / "reference/trained_policy/actor_weights.npz"

print(f"Loading {MODEL} ...")
model = SAC.load(str(MODEL), device="cpu")
actor = model.policy.actor

# Locate the MLP trunk — attribute name differs by SB3 version
if hasattr(actor, 'latent_pi'):
    mlp_trunk = actor.latent_pi          # SB3 2.x SAC Actor
elif hasattr(actor, 'mlp_extractor'):
    mlp_trunk = actor.mlp_extractor.policy_net  # older SB3
else:
    raise AttributeError(
        f"Cannot find MLP trunk in Actor. Attributes: {list(vars(actor).keys())}"
    )

# Collect Linear layers and activation names from the MLP trunk
mlp_weights, mlp_biases, act_names = [], [], []
for m in mlp_trunk:
    if isinstance(m, torch.nn.Linear):
        mlp_weights.append(m.weight.detach().float().numpy())
        mlp_biases.append( m.bias.detach().float().numpy())
    else:
        act_names.append(type(m).__name__)

mu_w = actor.mu.weight.detach().float().numpy()
mu_b = actor.mu.bias.detach().float().numpy()

n = len(mlp_weights)
arrays = {}
arrays["n_layers"]   = np.array(n)
arrays["act_names"]  = np.array(act_names)       # e.g. ['ReLU', 'ReLU']
for i in range(n):
    arrays[f"mlp_w{i}"] = mlp_weights[i]
    arrays[f"mlp_b{i}"] = mlp_biases[i]
arrays["mu_w"] = mu_w
arrays["mu_b"] = mu_b

np.savez(OUT, **arrays)

print(f"Saved → {OUT}")
print(f"n_layers={n}  obs_dim={mlp_weights[0].shape[1]}  act_dim={mu_w.shape[0]}")
for i, (w, b) in enumerate(zip(mlp_weights, mlp_biases)):
    print(f"  MLP layer {i}: Linear({w.shape[1]} → {w.shape[0]})  "
          f"activation={act_names[i] if i < len(act_names) else 'none'}")
print(f"  mu head:     Linear({mu_w.shape[1]} → {mu_w.shape[0]}) + tanh")
