import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from functools import partial
from inpaint_mask_func import draw_masks_from_boxes
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torch.cuda.amp import autocast
from trainer import read_official_ckpt, batch_to_device
from transformers import CLIPProcessor, CLIPModel
from reid_func_old import reid_loss, calc_acc
import wandb

device = "cuda"


def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas


def project(x, projection_matrix):
    return x@torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input == None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda() # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda()  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            feature = project(feature, torch.load('projection_matrix').cuda().T).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7 
            feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature


def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask


@torch.no_grad()
def prepare_batch(meta, batch=1, max_objs=30):
    phrases, images = meta.get("phrases"), meta.get("images")
    images = [None]*len(phrases) if images==None else images 
    phrases = [None]*len(images) if phrases==None else phrases 

    # version = "openai/clip-vit-large-patch14"
    version = "./gligen_checkpoints/clip-vit-large-patch14" #61's code
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)
    
    text_features = []
    image_features = []
    for phrase, image in zip(phrases,images):
        text_features.append(get_clip_feature(model, processor, phrase, is_image=False) )
        image_features.append(get_clip_feature(model, processor, image,  is_image=True) )

    for idx, (box, text_feature, image_feature) in enumerate(zip( meta['locations'], text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1 
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1 

    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_masks" : text_masks.unsqueeze(0).repeat(batch,1)*complete_mask(meta.get("text_mask"), max_objs),
        "image_masks" : image_masks.unsqueeze(0).repeat(batch,1)*complete_mask(meta.get("image_mask"), max_objs),
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
        "image_embeddings" : image_embeddings.unsqueeze(0).repeat(batch,1,1)
    }

    return batch_to_device(out, device) 


def load_ckpt(ckpt_path):
    
    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"], strict=False)
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,  default="generation_samples", help="root folder for output")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    parser.add_argument("--seed", type=int, default=123, help="random seed used in spsa")
    parser.add_argument("--spsa_o", type=float, default=0.0, help="hyparam in spsa: o")
    parser.add_argument("--spsa_c", type=float, default=0.001, help="hyparam in spsa: c")
    parser.add_argument("--spsa_a", type=float, default=40.0, help="hyparam in spsa: a")
    parser.add_argument("--spsa_alpha", type=float, default=0.6, help="hyparam in spsa: alpha")
    parser.add_argument("--spsa_gamma", type=float, default=0.1, help="hyparam in spsa: gamma")
    parser.add_argument("--spsa_b1", type=float, default=0.9, help="first moment scale in spsa")
    parser.add_argument("--spsa_avg", type=int, default=5, help="grad estimates averaging steps in spsa")
    parser.add_argument("--g_scale", type=int, default=0.000001, help="scale of the gradient")
    #parser.add_argument("--negative_prompt", type=str,  default=None, help="")
    args = parser.parse_args()
    return args


def get_meta():
    meta = dict(
        # ckpt = "../gligen_checkpoints/checkpoint_inpainting_text.pth",
        ckpt = "./gligen_checkpoints/diffusion_pytorch_model.bin",
        input_image = "inference_images/0034_c3s1_002826_01.jpg",
        prompt = "a bag",
        phrases =   ['bag'],
        locations = [ [0.25, 0.28, 0.42, 0.52] ], # mask will be derived from box 
        save_folder_name="inpainting_box_text/240116"
    )
    return meta


def get_gen_image(meta, diffusion, model, autoencoder, batch, config, context, uc, starting_noise=None):
    # print("start sanpler prepare")
    # - - - - - sampler - - - - - # 
    alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
    if config.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 250 
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 50 
    
    # - - - - - inpainting related - - - - - #
    # print("start inpainting prepare")
    inpainting_mask = z0 = None  # used for replacing known region in diffusion process
    inpainting_extra_input = None # used as model input 
    if "input_image" in meta:
        # inpaint mode 
        assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'
        
        inpainting_mask = draw_masks_from_boxes(batch['boxes'], model.image_size).cuda()
        
        input_image = TF.pil_to_tensor( Image.open(meta["input_image"]).convert("RGB").resize((512,512)) ) 
        input_image = ( input_image.float().unsqueeze(0).cuda() / 255 - 0.5 ) / 0.5
        z0 = autoencoder.encode( input_image )
        
        masked_z = z0 * inpainting_mask
        inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1) 
        
    # - - - - - input for gligen - - - - - #
    # print("start gligen input")
    grounding_input = grounding_tokenizer_input.prepare(batch)
    grounding_extra_input = None
    if grounding_downsampler_input != None:
        grounding_extra_input = grounding_downsampler_input.prepare(batch)
    input = dict(
        x = starting_noise, 
        timesteps = None, 
        context = context, 
        grounding_input = grounding_input,
        inpainting_extra_input = inpainting_extra_input,
        grounding_extra_input = grounding_extra_input,
    )
    
    # - - - - - start sampling - - - - - #
    # print("start sampling")
    shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)
    samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0)
    samples_fake = autoencoder.decode(samples_fake)
    # print(samples_fake[0])
    
    # - - - - - save - - - - - #
    # print("start save")
    output_folder = os.path.join( args.folder,  meta["save_folder_name"])
    os.makedirs( output_folder, exist_ok=True)
    start = len( os.listdir(output_folder) )
    image_ids = list(range(start,start+config.batch_size))
    # print(samples_fake[0].shape)
    samples_fake_reshape = [F.interpolate(samp.unsqueeze(0), size=(128, 64), mode='bilinear').squeeze(0) for samp in samples_fake]
    # print(image_ids)
    '''
    for image_id, sample in zip(image_ids, samples_fake_reshape):
        print(image_id, sample.shape)
        img_name = str(int(image_id))+'.png'
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().detach().numpy().transpose(1,2,0) * 255 
        sample = Image.fromarray(sample.astype(np.uint8))
        sample.save(  os.path.join(output_folder, img_name)   )
    
    '''
    return samples_fake_reshape[0]


def spsa_grad_estimate_bi(w, ck, avg, meta, diffusion, model, autoencoder, batch, config, uc, g_scale):
    ghats = []
    loss_fn = reid_loss
    acc_fn = None
    
    # print(w)
    
    avg = 10
    
    for spk in range(avg):
        print("spk:", spk)
        p_side = (torch.rand(w.numel()).reshape(w.shape) + 1) / 2
        samples = torch.cat([p_side, -p_side], dim=1)
        perturb = torch.gather(samples, 1, torch.bernoulli(torch.ones_like(p_side) / 2).type(torch.int64)).cuda()
        del samples
        del p_side
        ghats = torch.zeros_like(perturb).to(device)
        w_r = w + ck * perturb
        w_l = w - ck * perturb
        output_r = get_gen_image(meta, diffusion, model, autoencoder, batch, config, w_r, uc)
        output_l = get_gen_image(meta, diffusion, model, autoencoder, batch, config, w_l, uc)
        loss_r = loss_fn(output_r) * g_scale
        loss_l = loss_fn(output_l) * g_scale
        ghat = (loss_r - loss_l) / ((2 * ck) * perturb)
        del perturb
        ghats = ghats + ghat
        
    if avg == 1:
        pass
    else:
        # ghat = torch.cat(ghats, dim=0).mean(dim=0)
        ghat = ghats / avg
        # print("ghat.shape:", ghat.shape)
    loss = ((loss_r + loss_l) / 2)
    acc = None
    # acc = (acc_fn(output_r) + (acc_fn(output_l)) / 2).item()
    # ghat = (ghat - ghat.mean(dim=(0, 1, 2))) / ghat.std(dim=(0, 1, 2))
    return ghat, loss, acc


def context_vip(context, meta, diffusion, model, autoencoder, batch, config, uc, max_epoch=100):
    current_context = context.clone()
    
    min_loss = 1000000.0
    
    for epoch in range(max_epoch):
        # sample = get_gen_image(meta, diffusion, model, autoencoder, batch, config, current_context, uc)
        # print("sample:", sample.shape)
        
        # spsa hyparams
        spsa_o, spsa_c, spsa_a, spsa_alpha, spsa_gamma = config.spsa_o, config.spsa_c, config.spsa_a, config.spsa_alpha, config.spsa_gamma
        spsa_b1 = config.spsa_b1
        spsa_avg = config.spsa_avg
        g_scale = config.g_scale
        
        step = epoch + 1
        m1 = 0
        
        with torch.no_grad():
            with autocast(): #! for fast training
                ak = spsa_a / (step + spsa_o) ** spsa_alpha
                ck = spsa_c / (step ** spsa_gamma)
                ghat, loss, acc = spsa_grad_estimate_bi(current_context, ck, spsa_avg, meta, diffusion, model, autoencoder, batch, config, uc, g_scale)
                if step == 1:
                    m1 = ghat
                else:
                    m1 = spsa_b1 * m1 + ghat
                accum_ghat = ghat + spsa_b1 * m1
                current_context = current_context - ak * accum_ghat
                min_loss = min(min_loss, loss.item())
                print("step:", step, " loss:", loss.item(), " min_loss:", min_loss)
                wandb.log({"loss": loss.item()})
    
    return current_context


if __name__ == "__main__":
    wandb.init(project="ReID-EXP", name="202402262329", tags="blackvip")
    args = get_args()
    meta = get_meta()
    print("args, meta ok!")
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(meta["ckpt"])
    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input
    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])
    # - - - - - update config from args - - - - - # 
    print("start config!")
    config.update(vars(args))
    config = OmegaConf.create(config)
    batch = prepare_batch(meta, config.batch_size)
    #context: type = torch.Tensor, size = [config.batch_size, 77, 168]
    context = text_encoder.encode([meta["prompt"]] * config.batch_size)
    uc = text_encoder.encode(config.batch_size * [""])
    
    # get_gen_image(meta, autoencoder, config, context)
    
    final_context = context_vip(context, meta, diffusion, model, autoencoder, batch, config, uc, 
                                max_epoch=10000)