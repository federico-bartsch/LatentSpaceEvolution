# Latent Space Evolution

This project implements an **evolutionary search algorithm** to generate images that match a textual description using **StyleGAN2-ADA** and **OpenAI CLIP**. The system evolves a population of latent vectors over multiple generations to maximize the similarity between generated images and a target text prompt.

---

## Features

- Uses **StyleGAN2-ADA** for high-quality face image generation (FFHQ pretrained model).  
- Uses **CLIP (ViT-L/14)** to evaluate image-text similarity as a fitness function.  
- Implements a **genetic algorithm** with crossover, mutation, and population diversity tracking.  
- Generates a **video** showing evolution across generations.  
- Plots metrics like **best/average fitness** and **population diversity** over generations.

---

## Usage

1. **Download the pretrained FFHQ model**:

   Place the `ffhq.pkl` file in the `models/` directory.  
   You can download it from the [StyleGAN2-ADA pretrained models](https://github.com/NVlabs/stylegan2-ada-pytorch#pre-trained-networks).

2. **Set the target prompt**:

   Open `latent_space_evolution.py` and set the `TARGET_TEXT` variable to your desired description.  

3. **Run the evolution script**:

```bash
python latent_space_evolution.py
