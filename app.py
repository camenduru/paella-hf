import gradio as gr
import torch
import open_clip
import torchvision
from huggingface_hub import hf_hub_download
from PIL import Image
from open_clip import tokenizer
from Paella.utils.modules import Paella
from arroz import Diffuzz, PriorModel
from transformers import AutoTokenizer, T5EncoderModel
from Paella.src.vqgan import VQModel
from Paella.utils.alter_attention import replace_attention_layers

model_repo = "dome272/Paella"
model_file = "paella_v3.pt"
prior_file = "prior_v1.pt"
vqgan_file = "vqgan_f4.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 4
latent_shape = (batch_size, 64, 64)  # latent shape of the generated image, we are using an f4 vqgan and thus sampling 64x64 will result in 256x256
prior_timesteps, prior_cfg, prior_sampler, clip_embedding_shape = 60, 3.0, "ddpm", (batch_size, 1024)

generator_timesteps = 12
generator_cfg = 5
prior_timesteps = 60
prior_cfg = 3.0
prior_sampler = 'ddpm'
clip_embedding_shape = (batch_size, 1024)


def to_pil(images):
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = (images * 255).round().astype("uint8")
    images = [Image.fromarray(image) for image in images]
    return images
    
def log(t, eps=1e-20):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)

def sample(model, c, x=None, negative_embeddings=None, mask=None, T=12, size=(32, 32), starting_t=0, temp_range=[1.0, 1.0], typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=-1, renoise_steps=11, renoise_mode='start'):
    with torch.inference_mode():
        r_range = torch.linspace(0, 1, T+1)[:-1][:, None].expand(-1, c.size(0)).to(c.device)
        temperatures = torch.linspace(temp_range[0], temp_range[1], T)
        preds = []
        if x is None:
            x = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
        elif mask is not None:
            noise = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
            x = noise * mask + (1-mask) * x
        init_x = x.clone()
        for i in range(starting_t, T):
            if renoise_mode == 'prev':
                prev_x = x.clone()
            r, temp = r_range[i], temperatures[i]
            logits = model(x, c, r)
            if classifier_free_scale >= 0:
                if negative_embeddings is not None:
                    logits_uncond = model(x, negative_embeddings, r)
                else:
                    logits_uncond = model(x, torch.zeros_like(c), r)
                logits = torch.lerp(logits_uncond, logits, classifier_free_scale)
            x = logits
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
            if typical_filtering:
                x_flat_norm = torch.nn.functional.log_softmax(x_flat, dim=-1)
                x_flat_norm_p = torch.exp(x_flat_norm)
                entropy = -(x_flat_norm * x_flat_norm_p).nansum(-1, keepdim=True)

                c_flat_shifted = torch.abs((-x_flat_norm) - entropy)
                c_flat_sorted, x_flat_indices = torch.sort(c_flat_shifted, descending=False)
                x_flat_cumsum = x_flat.gather(-1, x_flat_indices).softmax(dim=-1).cumsum(dim=-1)

                last_ind = (x_flat_cumsum < typical_mass).sum(dim=-1)
                sorted_indices_to_remove = c_flat_sorted > c_flat_sorted.gather(1, last_ind.view(-1, 1))
                if typical_min_tokens > 1:
                    sorted_indices_to_remove[..., :typical_min_tokens] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, x_flat_indices, sorted_indices_to_remove)
                x_flat = x_flat.masked_fill(indices_to_remove, -float("Inf"))
            x_flat = torch.multinomial(x_flat.div(temp).softmax(-1), num_samples=1)[:, 0]
            x = x_flat.view(x.size(0), *x.shape[2:])
            if mask is not None:
                x = x * mask + (1-mask) * init_x
            if i < renoise_steps:
                if renoise_mode == 'start':
                    x, _ = model.add_noise(x, r_range[i+1], random_x=init_x)
                elif renoise_mode == 'prev':
                    x, _ = model.add_noise(x, r_range[i+1], random_x=prev_x)
                else: # 'rand'
                    x, _ = model.add_noise(x, r_range[i+1])
            preds.append(x.detach())
    return preds

# Model loading

# Load T5 on CPU
t5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-xl")
t5_model = T5EncoderModel.from_pretrained("google/byt5-xl")

# Load other models on GPU
clip_model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
clip_model = clip_model.to(device).half().eval().requires_grad_(False)

clip_preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
])

vqgan_path = hf_hub_download(repo_id=model_repo, filename=vqgan_file)
vqmodel = VQModel().to(device)
vqmodel.load_state_dict(torch.load(vqgan_path, map_location=device))
vqmodel.eval().requires_grad_(False)

prior_path = hf_hub_download(repo_id=model_repo, filename=prior_file)
prior = PriorModel().to(device)#.half()
prior.load_state_dict(torch.load(prior_path, map_location=device))
prior.eval().requires_grad_(False)

model_path = hf_hub_download(repo_id=model_repo, filename=model_file)
model = Paella(byt5_embd=2560)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().requires_grad_()#.half()
replace_attention_layers(model)
model.to(device)

diffuzz = Diffuzz(device=device)

@torch.inference_mode()
def decode(img_seq):
    return vqmodel.decode_indices(img_seq)

@torch.inference_mode()
def embed_t5(text, t5_tokenizer, t5_model, final_device="cuda"):
    device = t5_model.device
    t5_tokens = t5_tokenizer(text, padding="longest", return_tensors="pt", max_length=768, truncation=True).input_ids.to(device)
    t5_embeddings = t5_model(input_ids=t5_tokens).last_hidden_state.to(final_device)
    return t5_embeddings    

@torch.inference_mode()
def sample(model, model_inputs, latent_shape,
           unconditional_inputs=None, init_x=None, steps=12, renoise_steps=None,
           temperature = (0.7, 0.3), cfg=(8.0, 8.0),
           mode = 'multinomial',        # 'quant', 'multinomial', 'argmax'
           t_start=1.0, t_end=0.0,
           sampling_conditional_steps=None, sampling_quant_steps=None, attn_weights=None
    ):
    device = unconditional_inputs["byt5"].device
    if sampling_conditional_steps is None:
        sampling_conditional_steps = steps
    if sampling_quant_steps is None:
        sampling_quant_steps = steps
    if renoise_steps is None:
        renoise_steps = steps-1
    if unconditional_inputs is None:
        unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}

    init_noise = torch.randint(0, model.num_labels, size=latent_shape, device=device)
    if init_x != None:
        sampled = init_x
    else:
        sampled = init_noise.clone()
    t_list = torch.linspace(t_start, t_end, steps+1)
    temperatures = torch.linspace(temperature[0], temperature[1], steps)
    cfgs = torch.linspace(cfg[0], cfg[1], steps)
    for i, tv in enumerate(t_list[:steps]):
        if i >= sampling_quant_steps:
            mode = "quant"
        t = torch.ones(latent_shape[0], device=device) * tv

        logits = model(sampled, t, **model_inputs, attn_weights=attn_weights)
        if cfg is not None and i < sampling_conditional_steps:
            logits = logits * cfgs[i] + model(sampled, t, **unconditional_inputs) * (1-cfgs[i])
        scores = logits.div(temperatures[i]).softmax(dim=1)

        if mode == 'argmax':
            sampled = logits.argmax(dim=1)
        elif mode == 'multinomial':
            sampled = scores.permute(0, 2, 3, 1).reshape(-1, logits.size(1))
            sampled = torch.multinomial(sampled, 1)[:, 0].view(logits.size(0), *logits.shape[2:])
        elif mode == 'quant':
            sampled = scores.permute(0, 2, 3, 1) @ vqmodel.vquantizer.codebook.weight.data
            sampled = vqmodel.vquantizer.forward(sampled, dim=-1)[-1]
        else:
            raise Exception(f"Mode '{mode}' not supported, use: 'quant', 'multinomial' or 'argmax'")

        if i < renoise_steps:
            t_next = torch.ones(latent_shape[0], device=device) * t_list[i+1]
            sampled = model.add_noise(sampled, t_next, random_x=init_noise)[0]
    return sampled

# -----

def infer(prompt, negative_prompt, seed):
    torch.manual_seed(seed)
    text = tokenizer.tokenize([prompt] * latent_shape[0]).to(device)
    with torch.inference_mode():
        if negative_prompt:
            clip_text_tokens_uncond = tokenizer.tokenize([negative_prompt] * len(text)).to(device)
            t5_embeddings_uncond = embed_t5([negative_prompt] * len(text), t5_tokenizer, t5_model)
        else:
            clip_text_tokens_uncond = tokenizer.tokenize([""] * len(text)).to(device)
            t5_embeddings_uncond = embed_t5([""] * len(text), t5_tokenizer, t5_model)

        t5_embeddings = embed_t5([prompt] * latent_shape[0], t5_tokenizer, t5_model)
        clip_text_embeddings = clip_model.encode_text(text)
        clip_text_embeddings_uncond = clip_model.encode_text(clip_text_tokens_uncond)

        with torch.autocast(device_type="cuda"):
            clip_image_embeddings = diffuzz.sample(
                prior, {'c': clip_text_embeddings}, clip_embedding_shape,
                timesteps=prior_timesteps, cfg=prior_cfg, sampler=prior_sampler
            )[-1]
                
            attn_weights = torch.ones((t5_embeddings.shape[1]))
            attn_weights[-4:] = 0.4  # reweigh attention weights for image embeddings --> less influence
            attn_weights[:-4] = 1.2  # reweigh attention weights for the rest --> more influence
            attn_weights = attn_weights.to(device)
        
            sampled_tokens = sample(model,
                                    model_inputs={'byt5': t5_embeddings, 'clip': clip_text_embeddings, 'clip_image': clip_image_embeddings}, unconditional_inputs={'byt5': t5_embeddings_uncond, 'clip': clip_text_embeddings_uncond, 'clip_image': None},
                                    temperature=(1.2, 0.2), cfg=(8,8), steps=32, renoise_steps=26, latent_shape=latent_shape, t_start=1.0, t_end=0.0,
                                    mode="multinomial", sampling_conditional_steps=20, attn_weights=attn_weights)
            
    sampled = decode(sampled_tokens)
    return to_pil(sampled.clamp(0, 1))
    
css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
        }
        #share-btn * {
            all: unset;
        }
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
"""

block = gr.Blocks(css=css)

with block:
    gr.HTML(
        f"""
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Paella Demo
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                Paella is a novel text-to-image model that uses a compressed quantized latent space, based on a VQGAN, and a masked training objective to achieve fast generation in ~10 inference steps. 

                This version builds on top of our initial paper, bringing Paella to a similar level as other state-of-the-art models, while preserving the compactness and clarity of the previous implementations. Please, refer to the resources below for details.
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                with gr.Column():
                    text = gr.Textbox(
                        label="Enter your prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="an image of a shiba inu, donning a spacesuit and helmet, traversing the uncharted terrain of a distant, extraterrestrial world, as a symbol of the intrepid spirit of exploration and the unrelenting curiosity that drives humanity to push beyond the bounds of the known",
                        elem_id="prompt-text-input",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                    negative = gr.Textbox(
                        label="Enter your negative prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="low quality, low resolution, bad image, blurry, blur",
                        elem_id="negative-prompt-text-input",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                    full_width=False,
                )

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

        with gr.Group():
            with gr.Accordion("Advanced settings", open=False):
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=2147483647,
                    step=1,
                    randomize=True,
                )

        text.submit(infer, inputs=[text, negative, seed], outputs=gallery)
        btn.click(infer, inputs=[text, negative, seed], outputs=gallery)

        gr.HTML(
            """
                <div class="footer">
                </div>
                <div class="acknowledgments">
                    <p><h4>Resources</h4>
                    <a href="https://arxiv.org/abs/2211.07292" style="text-decoration: underline;">Paper</a>, <a href="https://github.com/dome272/Paella" style="text-decoration: underline;">official implementation</a>, <a href="https://huggingface.co/dome272/Paella" style="text-decoration: underline;">Model Card</a>.
                    </p>
                    <p><h4>LICENSE</h4>
                    <a href="https://github.com/dome272/Paella/blob/main/LICENSE" style="text-decoration: underline;">MIT</a>.
                    </p>
                    <p><h4>Biases and content acknowledgment</h4>
Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence. The model was trained on 600 million images from the improved <a href="https://laion.ai/blog/laion-5b/" style="text-decoration: underline;" target="_blank">LAION-5B aesthetic</a> dataset, which scraped non-curated image-text-pairs from the internet (the exception being the removal of illegal content) and is meant for research purposes.
                    </p>
               </div>
           """
        )

block.launch()
