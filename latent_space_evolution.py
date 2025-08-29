import os
import torch
import random
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as TF
from transformers import CLIPProcessor, CLIPModel


import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
sys.path.append("./stylegan2-ada-pytorch")
import legacy
import dnnlib
def population_diversity_cosine(Z: torch.Tensor) -> float:
    # Normalize latent vectors to unit length
    Z_norm = F.normalize(Z, p=2, dim=1)
    
    # Compute cosine similarity matrix
    cos_sim_matrix = Z_norm @ Z_norm.T
    
    # Get upper triangle indices excluding diagonal
    N = Z.shape[0]
    triu_indices = torch.triu_indices(N, N, offset=1)
    
    # Average pairwise cosine similarity
    avg_cos_sim = cos_sim_matrix[triu_indices[0], triu_indices[1]].mean()
    
    # Diversity = 1 - average similarity
    diversity = 1 - avg_cos_sim
    return diversity.item()

# ------------------- Setup -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load StyleGAN2-ADA FFHQ model
print("Loading StyleGAN2-ADA FFHQ model...")
with open('/home/federico.bartsch/bio_model/ffhq.pkl', 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

# Load CLIP ViT-L/14
print("Loading CLIP ViT-L/14 model...")
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Hyperparameters
POP_SIZE = 20
GENERATIONS = 100
mut_std = 0.5
TARGET_TEXT = "An asian guy wearing a hat and glasses"
z_dim = G.z_dim

# Create frames directory
frames_path = os.path.join(os.path.dirname(__file__), "video_frames")
video_path = os.path.join(os.path.dirname(__file__), "video", "evolution.mp4")
frames = []

# Stats for plotting
best_scores_per_gen = []
avg_scores_per_gen = []
diversity_per_gen = []
mutation_events = {"prob": [], "diversity": []}

# ------------------- Functions -------------------
def generate_image(z):
    with torch.no_grad():
        img = G(z.unsqueeze(0), None, truncation_psi=0.8, noise_mode='const')
    img = (img + 1) / 2
    return img.clamp(0, 1)

def fitness(img_tensor, target=TARGET_TEXT):
    img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    image_inputs = processor(images=pil_img, return_tensors="pt").to(device)
    text_inputs = processor(text=target, return_tensors="pt").to(device)
    with torch.no_grad():
        img_feat = clip.get_image_features(**image_inputs)
        txt_feat = clip.get_text_features(**text_inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    return torch.cosine_similarity(img_feat, txt_feat).item()

# ------------------- Initialize population -------------------
population = [torch.randn(z_dim, device=device) for _ in range(POP_SIZE)]
best_score = -1
best_z = None

# ------------------- Evolution loop -------------------
print("Starting evolutionary search...")
for gen in range(GENERATIONS):
    scored_population = []
    images = []

    # Evaluate population
    for z in population:
        img = generate_image(z)
        score = fitness(img)
        scored_population.append((score, z, img))
        images.append(img)
    

    # Track stats
    scores = [s for s, _, _ in scored_population]
    best_scores_per_gen.append(max(scores))
    avg_scores_per_gen.append(np.mean(scores))
    
    Z = torch.stack([z for _, z, _ in scored_population], dim=0)
    diversity_per_gen.append(population_diversity_cosine(Z))
    # Sort by score
    scored_population.sort(key=lambda x: x[0], reverse=True)
    top_score, top_z, top_img = scored_population[0]

    if top_score > best_score:
        best_score = top_score
        best_z = top_z
        #save_image(top_img, os.path.join(os.path.dirname(__file__), "images", f"best_gen{gen}_score{top_score:.4f}.png"))
        print(f"[Gen {gen}] New best score: {top_score:.4f}")

    # Create grid
    imgs_tensor = torch.cat([img for _, _, img in scored_population], dim=0)
    grid = make_grid(imgs_tensor, nrow=5, padding=4)
    grid = TF.resize(grid, [512, 512])  # downscale for video

    grid_pil = TF.to_pil_image(grid)
    draw = ImageDraw.Draw(grid_pil)

    # Highlight best image
    nrow = 5
    thumb_w = grid_pil.width // nrow
    thumb_h = grid_pil.height // (len(images) // nrow)
    best_idx = 0
    row, col = divmod(best_idx, nrow)
    x0, y0 = col * thumb_w, row * thumb_h
    x1, y1 = x0 + thumb_w, y0 + thumb_h
    draw.rectangle([x0, y0, x1, y1], outline="red", width=4)

    # Save frame
    frame_path = os.path.join(frames_path, f"gen_{gen:04d}.png")
    grid_pil.save(frame_path)
    frames.append(imageio.imread(frame_path))

    # Evolution step
    new_population = [top_z.clone()]
    top_k = [z for _, z, _ in scored_population[:POP_SIZE // 2]]
    while len(new_population) < POP_SIZE:
        parent1, parent2 = random.sample(top_k, 2)
        child = 0.5 * (parent1 + parent2)

        if diversity_per_gen[-1] < 0.3:
            if random.random() < 0.5:
                print("Low diversity detected, applying stronger mutation.")
                child += torch.randn_like(child) * (2 * mut_std)
                mutation_events["diversity"].append(gen+1)
        elif random.random() < 0.2:
            child += torch.randn_like(child) * (2 * mut_std)
            mutation_events["prob"].append(gen+1)
            print("Probability-based mutation applied.")
        else:
            child += torch.randn_like(child) * mut_std
        new_population.append(child)
    population = new_population

# ------------------- Create video -------------------
imageio.mimsave(video_path, frames, fps=1)
print(f"Video saved at {video_path}")

# ------------------- Plot Fitness -------------------
plt.figure(figsize=(10,5))
plt.plot(best_scores_per_gen, label="Best Fitness", color='blue', linewidth=2)
plt.plot(avg_scores_per_gen, label="Average Fitness", color='orange', linestyle='--', linewidth=2)
plt.xlabel("Generation")
plt.ylabel("CLIP Similarity")
plt.title("Best vs Average Fitness per Generation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "evolution_best_avg.png"), dpi=200)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(diversity_per_gen, label="Population Diversity", color='blue', linestyle='-.', linewidth=2)
plt.xlabel("Generation")
plt.ylabel("Value")
plt.title("Diversity over Generations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "evolution_diversity.png"), dpi=200)
plt.show()

# plt.figure(figsize=(10,5))
# plt.plot(diversity_per_gen, label="Population Diversity", color='blue', linestyle='-.', linewidth=2)

# # Overlay mutation events as scatter points
# plt.scatter(mutation_events["prob"], 
#             [diversity_per_gen[g] for g in mutation_events["prob"]], 
#             color="green", marker="o", s=60, label="Prob. mutation")

# plt.scatter(mutation_events["diversity"], 
#             [diversity_per_gen[g] for g in mutation_events["diversity"]], 
#             color="red", marker="x", s=80, label="Diversity mutation")

# plt.xlabel("Generation")
# plt.ylabel("Diversity")
# plt.title("Diversity over Generations with Mutation Events")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "evolution_diversity_mutations.png"), dpi=200)
# plt.show()

print("Evolution complete.")
print(mutation_events)
