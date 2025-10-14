# story_to_image.py
import os
import torch
from transformers import pipeline, set_seed
from diffusers import StableDiffusionPipeline
from PIL import Image
import textwrap

# ========== CONFIG ==========
# Choose text model: "gpt2" is small and downloads automatically.
TEXT_MODEL = "gpt2"
# Choose stable diffusion model from HF hub (ensure you accepted license and have token if needed)
SD_MODEL = "runwayml/stable-diffusion-v1-5"  # change if you prefer stable-diffusion-xl or others

# If GPU is available, use it
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Generation defaults
TEXT_MAX_NEW_TOKENS = 200
TEXT_TEMPERATURE = 0.9
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
OUTPUT_IMAGE_PATH = "generated_story_image.png"

# ========== HELPERS ==========
def make_story(prompt: str) -> str:
    """Generate a short creative story from user's prompt using a text-generation pipeline."""
    gen = pipeline("text-generation", model=TEXT_MODEL, device=0 if DEVICE=="cuda" else -1)
    set_seed(42)  # for reproducibility (optional)
    out = gen(prompt,
              max_new_tokens=TEXT_MAX_NEW_TOKENS,
              do_sample=True,
              temperature=TEXT_TEMPERATURE,
              top_k=50,
              top_p=0.95,
              num_return_sequences=1)
    story = out[0]["generated_text"]
    return story.strip()

def extract_scene_line(story: str) -> str:
    """Simple heuristic: pick first long sentence describing a scene.
    You can replace this with a smarter summarizer or prompt-engineered extractor."""
    sentences = [s.strip() for s in story.replace("\n"," ").split(".") if s.strip()]
    # prefer sentences that are somewhat descriptive
    for s in sentences:
        if len(s.split()) > 6:
            return s + "."
    return sentences[0] + "." if sentences else story

def load_sd_pipeline(model_name: str):
    """Load Stable Diffusion pipeline. Requires model access for some models."""
    # If you have a HF token in environment, diffusers will use it automatically.
    # Use torch_dtype=torch.float16 for CUDA if available and if model supports it.
    if DEVICE == "cuda":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            revision="fp16" if "fp16" in torch.cuda.get_device_capability() or True else None,
            use_safetensors=True,
        )
        pipe = pipe.to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_name)
        pipe = pipe.to("cpu")

    # Optional: enable xformers memory-efficient attention (if installed)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe

def create_image_from_prompt(pipe, prompt: str, output_path: str):
    generator = torch.Generator(device=DEVICE) if DEVICE=="cuda" else None
    if generator is not None:
        generator = generator.manual_seed(42)
    images = pipe(prompt,
                  height=IMAGE_HEIGHT,
                  width=IMAGE_WIDTH,
                  num_inference_steps=NUM_INFERENCE_STEPS,
                  guidance_scale=GUIDANCE_SCALE,
                  generator=generator).images
    if images and len(images) > 0:
        img = images[0]
        img.save(output_path)
        return output_path
    return None

# ========== MAIN ==========
def main():
    print("Welcome to AI Story-to-Image Generator!\n")
    user_prompt = input("Type a short story prompt (e.g., 'a lonely lighthouse in a neon storm'): ").strip()
    if not user_prompt:
        print("Please provide a prompt. Exiting.")
        return

    print("\nGenerating a story from your prompt...\n")
    story = make_story(user_prompt)
    print("---- GENERATED STORY ----")
    print("\n".join(textwrap.wrap(story, width=100)))
    print("-------------------------\n")

    scene_line = extract_scene_line(story)
    print("Scene extracted for image generation:")
    print(scene_line, "\n")

    print(f"Loading Stable Diffusion model ({SD_MODEL}) — this may take a while the first time...")
    pipe = load_sd_pipeline(SD_MODEL)

    # You can enrich the prompt with style directives:
    style_suffix = "highly detailed, cinematic lighting, 4k, photorealistic"
    final_prompt = f"{scene_line} {style_suffix}"
    print("Final image prompt:\n", final_prompt, "\n")

    print("Generating image (this will use your GPU if available)...")
    out_path = create_image_from_prompt(pipe, final_prompt, OUTPUT_IMAGE_PATH)

    if out_path:
        print(f"Image saved to: {out_path}")
    else:
        print("Image generation failed.")

if __name__ == "__main__":
    main()
