# import torch
# from transformers import AutoTokenizer, pipeline
# from transformers import AutoModelForCausalLM
# from optimum.intel import IPEXModelForCausalLM

# model_id = "gpt2"
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# model = IPEXModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, export=True)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# results = pipe("He's a dreadful magician and")



# from transformers import AutoModelForCausalLM
# from optimum.exporters.openvino import export_from_model

# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
# export_from_model(model, output="ov_model", task="text-generation-with-past")


from diffusers import DiffusionPipeline
import torch

# Load Stable Diffusion XL Base1.0
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")


# Optional CPU offloading to save some GPU Memory
pipe.enable_model_cpu_offload()

# Loading Trained LoRA Weights
pipe.load_lora_weights("AdamLucek/sdxl-base-1.0-oldbookillustrations-lora")

prompt = "A Dachsund walks confidently down a dirt path"

# Invoke pipeline to generate image
image = pipe(
    prompt = prompt,
    num_inference_steps=5,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]

image.show()