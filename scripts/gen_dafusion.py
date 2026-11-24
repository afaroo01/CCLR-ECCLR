"""
DA-Fusion synthetic data for the PV -> PD single-source comparison (generative / diffusion augmentation family).

Trabucco, B., Doherty, K., Gurinas, M., Salakhutdinov, R., 2024. Effective data augmentation with diffusion models.
International Conference on Learning Representations (ICLR 2024). Official code: github.com/brandontrabucco/da-fusion

Follows the official settings:
  1. Textual inversion: one new token per class, '<class_name>', initialised from 'the'; Stable Diffusion v1.4;
     512 px; batch 4; 1000 steps; AdamW lr 5e-4 scaled by batch size (= 2e-3), weight decay 1e-2, constant schedule;
     imagenet_templates_small prompts; random horizontal flip. Only the new token embedding is learned.
     Inversion uses up to 16 images per class (the official few-shot maximum) from the PV TRAINING split only.
  2. Augmentation: img2img with prompt 'a photo of a <class_name>', strength 0.5, guidance 7.5, safety checker off.
     M variants per training image (--num-synthetic, default 1 here because PV is not few-shot).
  3. Training (dg_baselines.py --method dafusion): each real image is replaced by one of its variants with p = 0.5.

No PlantDoc image and no field-specific prompt is used, so the diffusion model only contributes its pretrained
(LAION) prior, as in the published method. Validation and test images are never used.

Run (from code_updated\\code\\scripts):
  pip install diffusers transformers accelerate
  python gen_dafusion.py                       # inversion (~1 h) + generation (~3 h for M = 1 on an RTX 5060)
  python dg_baselines.py --method dafusion --backbone effnet --seeds 0 1 2 3 4
The script resumes: finished embeddings and images are skipped.
"""
import argparse, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dg_baselines import MappedFolder, CLASSES, split_indices

TEMPLATES = ["a photo of a {}", "a rendering of a {}", "a cropped photo of the {}", "the photo of a {}",
             "a photo of a clean {}", "a photo of a dirty {}", "a dark photo of the {}", "a photo of my {}",
             "a photo of the cool {}", "a close-up photo of a {}", "a bright photo of the {}", "a cropped photo of a {}",
             "a photo of the {}", "a good photo of the {}", "a photo of one {}", "a close-up photo of the {}",
             "a rendition of the {}", "a photo of the clean {}", "a rendition of a {}", "a photo of a nice {}",
             "a good photo of a {}", "a photo of the nice {}", "a photo of the small {}", "a photo of the weird {}",
             "a photo of the large {}", "a photo of a cool {}", "a photo of a small {}"]   # imagenet_templates_small


def token_of(c):
    return f"<{c.lower().replace('___', '_')}>"


def to_tensor(img, size, flip):
    img = img.convert('RGB').resize((size, size), Image.BICUBIC)
    if flip and random.random() < 0.5: img = img.transpose(Image.FLIP_LEFT_RIGHT)
    a = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1)


def invert(args, dev, tok, te, vae, unet, sched, images_by_class):
    emb_path = os.path.join(args.out, 'embeddings.pt')
    learned = torch.load(emb_path) if os.path.exists(emb_path) else {}
    E = te.get_input_embeddings()
    init = E.weight.data[tok.encode('the', add_special_tokens=False)[0]].clone()
    for c in CLASSES:
        tid = tok.convert_tokens_to_ids(token_of(c))
        E.weight.data[tid] = learned[token_of(c)].to(E.weight.dtype) if token_of(c) in learned else init
    unet.enable_gradient_checkpointing(); unet.train(); te.train()
    for c in CLASSES:
        t = token_of(c)
        if t in learned: print(f'  inversion {t}: done earlier'); continue
        tid = tok.convert_tokens_to_ids(t)
        orig = E.weight.data.clone()
        opt = torch.optim.AdamW(E.parameters(), lr=args.lr * args.inv_batch, weight_decay=1e-2)
        paths = images_by_class[c]; t0 = time.time()
        for step in range(args.inv_steps):
            batch = torch.stack([to_tensor(Image.open(random.choice(paths)), args.res, True) for _ in range(args.inv_batch)]).to(dev)
            ids = tok([random.choice(TEMPLATES).format(t) for _ in range(args.inv_batch)], padding='max_length',
                      truncation=True, max_length=te.config.max_position_embeddings, return_tensors='pt').input_ids.to(dev)
            with torch.no_grad():
                lat = vae.encode(batch.to(vae.dtype)).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(lat)
            ts = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=dev).long()
            noisy = sched.add_noise(lat, noise, ts)
            with torch.autocast('cuda' if dev.type == 'cuda' else 'cpu', dtype=torch.float16 if dev.type == 'cuda' else torch.bfloat16):
                enc = te(ids)[0]
                pred = unet(noisy, ts, enc.to(unet.dtype)).sample
            loss = F.mse_loss(pred.float(), noise.float())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad():                                      # keep every other embedding fixed (official)
                keep = torch.arange(E.weight.shape[0], device=E.weight.device) != tid
                E.weight.data[keep] = orig[keep]
            if step % 100 == 0: print(f'  inversion {t} step {step}: loss {loss.item():.4f} ({time.time() - t0:.0f} s)', flush=True)
        learned[t] = E.weight.data[tid].detach().cpu().clone()
        torch.save(learned, emb_path)
    unet.eval(); te.eval()
    return learned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pv', nargs='+', default=['C:/Adil PhD Work/WP2 CCLR_ECCLR/data/plantvillage'])
    ap.add_argument('--out', default='C:/Adil PhD Work/WP2 CCLR_ECCLR/data/plantvillage_dafusion')
    ap.add_argument('--model-path', default='CompVis/stable-diffusion-v1-4')
    ap.add_argument('--examples-per-class', type=int, default=16)
    ap.add_argument('--inv-steps', type=int, default=1000)
    ap.add_argument('--inv-batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--num-synthetic', type=int, default=1, help='variants per training image (M)')
    ap.add_argument('--strength', type=float, default=0.5)
    ap.add_argument('--guidance', type=float, default=7.5)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--gen-batch', type=int, default=4)
    ap.add_argument('--save-size', type=int, default=256, help='saved resolution (PV images are 256 px)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='debug: only the first N training images')
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fp = torch.float16 if dev.type == 'cuda' else torch.float32
    os.makedirs(args.out, exist_ok=True)

    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionImg2ImgPipeline
    from transformers import CLIPTokenizer, CLIPTextModel

    full = MappedFolder(args.pv)
    tr_idx, _, _ = split_indices(len(full))
    train = [full.samples[i] for i in tr_idx]
    if args.limit: train = train[:args.limit]
    by_class = {c: [p for p, y in train if CLASSES[y] == c] for c in CLASSES}
    rng = random.Random(args.seed)
    inv_imgs = {c: rng.sample(v, min(args.examples_per_class, len(v))) for c, v in by_class.items()}
    print(f'  {len(train)} training images; inversion uses {args.examples_per_class} per class')

    tok = CLIPTokenizer.from_pretrained(args.model_path, subfolder='tokenizer')
    te = CLIPTextModel.from_pretrained(args.model_path, subfolder='text_encoder').to(dev)
    tok.add_tokens([token_of(c) for c in CLASSES]); te.resize_token_embeddings(len(tok))
    for p in te.parameters(): p.requires_grad_(False)
    te.get_input_embeddings().weight.requires_grad_(True)
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder='vae', torch_dtype=fp).to(dev)
    unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder='unet', torch_dtype=fp).to(dev)
    for m in (vae, unet):
        for p in m.parameters(): p.requires_grad_(False)
    sched = DDPMScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    invert(args, dev, tok, te, vae, unet, sched, inv_imgs)

    te.get_input_embeddings().weight.requires_grad_(False)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model_path, vae=vae, unet=unet, text_encoder=te.to(fp),
                                                          tokenizer=tok, torch_dtype=fp, safety_checker=None,
                                                          requires_safety_checker=False).to(dev)
    pipe.set_progress_bar_config(disable=True)
    jobs = []
    for p, y in train:
        c = CLASSES[y]; stem = os.path.splitext(os.path.basename(p))[0]
        os.makedirs(os.path.join(args.out, c), exist_ok=True)
        for k in range(args.num_synthetic):
            dst = os.path.join(args.out, c, f'{stem}__{k}.png')
            if not os.path.exists(dst): jobs.append((p, c, dst))
    print(f'  generating {len(jobs)} images (strength {args.strength}, guidance {args.guidance})', flush=True)
    g = torch.Generator(device=dev).manual_seed(args.seed); t0 = time.time()
    for b in range(0, len(jobs), args.gen_batch):
        chunk = jobs[b:b + args.gen_batch]
        imgs = [Image.open(p).convert('RGB').resize((args.res, args.res), Image.BILINEAR) for p, _, _ in chunk]
        out = pipe(prompt=[f'a photo of a {token_of(c)}' for _, c, _ in chunk], image=imgs, strength=args.strength,
                   guidance_scale=args.guidance, num_inference_steps=args.steps, generator=g).images
        for im, (_, _, dst) in zip(out, chunk):
            im.resize((args.save_size, args.save_size), Image.BICUBIC).save(dst)
        if (b // args.gen_batch) % 50 == 0:
            done = b + len(chunk); el = time.time() - t0
            print(f'  {done}/{len(jobs)} images, {el / 60:.1f} min, about {el / done * (len(jobs) - done) / 60:.0f} min left', flush=True)
    print('  done:', args.out)


if __name__ == '__main__':
    main()
