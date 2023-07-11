# Prediction interface for Cog ⚙️
import cog
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Import other necessary modules here...
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler

class Predictor(cog.Predictor):
    def setup(self):
        self.args = {
            "pretrained_model_path": "models/StableDiffusion/stable-diffusion-v1-5",
            "inference_config": "configs/inference/inference.yaml",
            "config": "path/to/your/config.yaml",  # replace with your actual config path
            "L": 16,
            "W": 512,
            "H": 512
        }
        
        self.config = OmegaConf.load(self.args["config"])

        # Load models and setup the pipeline
        inference_config = OmegaConf.load(self.args["inference_config"])
        tokenizer    = CLIPTokenizer.from_pretrained(self.args["pretrained_model_path"], subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.args["pretrained_model_path"], subfolder="text_encoder")
        vae          = AutoencoderKL.from_pretrained(self.args["pretrained_model_path"], subfolder="vae")            
        unet         = UNet3DConditionModel.from_pretrained_2d(self.args["pretrained_model_path"], subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

        self.pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

    @cog.input("prompt", type=str, help="Prompt for the animation")
    def predict(self, prompt):
        # Set the prompt in the config
        for config_key, model_config in self.config.items():
            model_config.prompt = [prompt]

        # Run the pipeline and generate the sample animation
        sample = self.pipeline(
            prompt,
            negative_prompt=n_prompt,  # replace with your actual value
            num_inference_steps=model_config.steps,
            guidance_scale=model_config.guidance_scale,
            width=self.args["W"],
            height=self.args["H"],
            video_length=self.args["L"],
        ).videos

        # Save the sample animation as a GIF
        output_path = Path("/tmp/output.gif")
        save_videos_grid(sample, str(output_path))

        return output_path
