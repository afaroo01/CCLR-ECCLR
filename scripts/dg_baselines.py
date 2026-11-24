"""
DG baselines for the CCLR/ECCLR revision (Reviewer #1, comments 4.3, 6.3, 7.4).

Single-source domain generalisation: train on PlantVillage (PV, 8 classes) only, evaluate on PlantDoc (PD).
Uses the same pipeline as exp1/exp5: same 8 classes, 70/20/10 split (train_test_split, random_state=42),
AutoAugment training transform, batch 32, up to 25 epochs, Adam (EfficientNet-B0) / AdamW (ViT-Tiny).

One method per DG family (from the DG SOTA report):
  erm       sanity check, should reproduce exp1 / exp5
  mixstyle  augmentation / style randomisation  (MixStyle, ICLR'21; token-level variant for ViT, as in TFS-ViT)
  swad      flat minima / weight averaging      (SWAD, NeurIPS'21; window chosen on source validation loss)
  miro      pretrained regularisation / KD      (MIRO, ECCV'22; regularises final features towards the frozen ImageNet model)
  clipood   vision-language (CLIP)              (CLIPood, ICML'23; margin metric softmax + Beta moving average; CLIP ViT-B/16)

Simplifications versus the original papers are listed in README_DG.md and must be stated in the paper.

Input folders are the ones your notebooks build with make_ds():
  --pv  .../data/train   (PV, 8 class subfolders, images resized to 224x224)
  --pd  .../data/test    (PD train+test merged, 8 class subfolders)

Example:
  python dg_baselines.py --pv C:/data/train --pd C:/data/test --method swad --backbone effnet --seeds 0 1 2 3 4
Results are appended to dg_results.csv (one row per run) and summarised with --summarise.
"""
import argparse, copy, csv, json, math, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# ----------------------------------------------------------------------------- data
def make_transforms(clip=False, augmix=False, plain=False):
    mean, std = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)) if clip else (MEAN, STD)
    if plain:                                        # ViT-MoE: processor only (resize 224, mean = std = 0.5), no augmentation
        t = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((224, 224)), v2.Normalize([0.5] * 3, [0.5] * 3)])
        return t, t
    policy = v2.AugMix() if augmix else v2.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.IMAGENET)
    train = v2.Compose([v2.ToImage(), policy, v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=mean, std=std),
                        v2.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(1.0, 1.0))])
    ev = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                     v2.Normalize(mean=mean, std=std), v2.Resize((224, 224))])
    return train, ev

CLASSES = ['Apple___healthy', 'Apple___rust', 'Bell_pepper___bacterial_spot', 'Bell_pepper___healthy',
           'Grape___black_rot', 'Grape___healthy', 'Tomato___healthy', 'Tomato___septoria_leaf_spot']
# same folder -> class maps as the notebooks (plant_village_map, plantdoc_map); canonical names map to themselves
FOLDER_MAP = {
    'Apple___healthy': 'Apple___healthy', 'Apple___Cedar_apple_rust': 'Apple___rust',
    'Pepper,_bell___Bacterial_spot': 'Bell_pepper___bacterial_spot', 'Pepper,_bell___healthy': 'Bell_pepper___healthy',
    'Grape___Black_rot': 'Grape___black_rot', 'Grape___healthy': 'Grape___healthy',
    'Tomato___healthy': 'Tomato___healthy', 'Tomato___Septoria_leaf_spot': 'Tomato___septoria_leaf_spot',
    'Apple_leaf': 'Apple___healthy', 'Apple_rust_leaf': 'Apple___rust', 'Bell_pepper_leaf': 'Bell_pepper___healthy',
    'Bell_pepper_leaf_spot': 'Bell_pepper___bacterial_spot', 'grape_leaf': 'Grape___healthy',
    'grape_leaf_black_rot': 'Grape___black_rot', 'Tomato_leaf': 'Tomato___healthy',
    'Tomato_Septoria_leaf_spot': 'Tomato___septoria_leaf_spot'}
FOLDER_MAP.update({c: c for c in CLASSES})
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

class MappedFolder(torch.utils.data.Dataset):
    """Images from one or more roots; class sub-folders are mapped to the 8 canonical classes, other folders ignored.
    A root that has train/ and test/ sub-folders (PlantDoc) is expanded to both."""
    def __init__(self, roots, transform=None):
        from PIL import Image
        self.Image, self.transform, self.samples = Image, transform, []
        expanded = []
        for r in roots:
            subs = [os.path.join(r, d) for d in ('train', 'test') if os.path.isdir(os.path.join(r, d))]
            expanded += subs if subs else [r]
        for r in expanded:
            for folder in sorted(os.listdir(r)):
                p = os.path.join(r, folder)
                if not os.path.isdir(p) or folder not in FOLDER_MAP: continue
                y = CLASSES.index(FOLDER_MAP[folder])
                for f in sorted(os.listdir(p)):
                    if os.path.splitext(f)[1].lower() in IMG_EXT: self.samples.append((os.path.join(p, f), y))
        counts = [sum(1 for _, y in self.samples if y == k) for k in range(len(CLASSES))]
        missing = [c for c, n in zip(CLASSES, counts) if n == 0]
        if missing: raise SystemExit(f'No images for {missing} under {expanded}. Check the folder path.')
        print(f'  {len(self.samples)} images from {expanded}: ' + ', '.join(f'{c}={n}' for c, n in zip(CLASSES, counts)))
        self.classes = CLASSES
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]
        img = self.Image.open(p).convert('RGB')
        return (self.transform(img) if self.transform else img), y

class SynMix(torch.utils.data.Dataset):
    """DA-Fusion training (Trabucco et al., ICLR 2024): each real training image is replaced, with probability
    p = 0.5, by one of its diffusion-generated variants (syn_dir/<class>/<stem>__<k>.png)."""
    def __init__(self, base, idx, transform, syn_dir, p=0.5):
        from PIL import Image
        self.base, self.idx, self.t, self.p, self.Image = base, list(idx), transform, p, Image
        self.syn = {}
        for c in os.listdir(syn_dir):
            d = os.path.join(syn_dir, c)
            if not os.path.isdir(d): continue
            for f in os.listdir(d):
                self.syn.setdefault((c, f.rsplit('__', 1)[0]), []).append(os.path.join(d, f))
        hit = sum(1 for i in self.idx if self._key(i) in self.syn)
        print(f'  DA-Fusion: synthetic variants found for {hit}/{len(self.idx)} training images')
        if hit == 0: raise SystemExit('No synthetic images match the training split: check --syn-dir')
    def _key(self, i):
        p, y = self.base.samples[i]
        return CLASSES[y], os.path.splitext(os.path.basename(p))[0]
    def __len__(self): return len(self.idx)
    def __getitem__(self, j):
        i = self.idx[j]; p, y = self.base.samples[i]
        cands = self.syn.get(self._key(i))
        if cands and random.random() < self.p: p = random.choice(cands)
        return self.t(self.Image.open(p).convert('RGB')), y

def split_indices(n):
    """70/20/10 split with sklearn train_test_split, random_state 42, identical to the notebooks."""
    n_tr = int(0.7 * n); n_val = int(0.2 * n); n_te = n - n_tr - n_val
    tr_idx, tmp = train_test_split(range(n), test_size=(n_val + n_te), random_state=42)
    val_idx, te_idx = train_test_split(tmp, test_size=n_te, random_state=42)
    return tr_idx, val_idx, te_idx

def make_loaders(pv, pd, bs, workers, clip=False, augmix=False, syn_dir=None, plain=False):
    tr_t, ev_t = make_transforms(clip, augmix, plain)
    full = MappedFolder(pv)
    tr_idx, val_idx, te_idx = split_indices(len(full))
    def view(t):
        d = copy.copy(full); d.transform = t; return d
    tr, va, te = Subset(view(tr_t), tr_idx), Subset(view(ev_t), val_idx), Subset(view(ev_t), te_idx)
    if syn_dir: tr = SynMix(full, tr_idx, tr_t, syn_dir)
    tg = MappedFolder(pd, transform=ev_t)
    kw = dict(batch_size=bs, num_workers=workers, pin_memory=torch.cuda.is_available())
    return (DataLoader(tr, shuffle=True, drop_last=True, **kw), DataLoader(va, shuffle=False, **kw),
            DataLoader(te, shuffle=False, **kw), DataLoader(tg, shuffle=False, **kw), CLASSES)

# ----------------------------------------------------------------------------- MixStyle
class MixStyle(nn.Module):
    """MixStyle (Zhou et al., ICLR'21), random-shuffle mixing (single source). p=0.5, alpha=0.1 as in the paper.
    tokens=True mixes statistics over the token dimension of a ViT (B, N, C), as in TFS-ViT."""
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, tokens=False):
        super().__init__(); self.p, self.beta, self.eps, self.tokens = p, torch.distributions.Beta(alpha, alpha), eps, tokens
    def forward(self, x):
        if not self.training or random.random() > self.p: return x
        B = x.size(0)
        dims = (1,) if self.tokens else (2, 3)
        mu = x.mean(dim=dims, keepdim=True); var = x.var(dim=dims, keepdim=True)
        sig = (var + self.eps).sqrt(); mu, sig = mu.detach(), sig.detach()
        xn = (x - mu) / sig
        lam = self.beta.sample((B,) + (1,) * (x.dim() - 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]
        return xn * (sig * lam + sig2 * (1 - lam)) + (mu * lam + mu2 * (1 - lam))

# ----------------------------------------------------------------------------- backbones
class EffNet(nn.Module):
    def __init__(self, nc, pretrained=True, mixstyle=False):
        super().__init__()
        from torchvision.models import efficientnet_b0
        m = efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        self.features, self.avgpool = m.features, m.avgpool
        self.dropout = m.classifier[0]
        self.fc = nn.Linear(m.classifier[1].in_features, nc)
        self.feat_dim = m.classifier[1].in_features
        self.ms = MixStyle() if mixstyle else None
        self.ms_after = {2, 3, 4}           # early stages, as MixStyle inserts after the first residual stages
    def embed(self, x):
        for i, blk in enumerate(self.features):
            x = blk(x)
            if self.ms is not None and i in self.ms_after: x = self.ms(x)
        return torch.flatten(self.avgpool(x), 1)
    def forward(self, x, return_feat=False):
        f = self.embed(x); out = self.fc(self.dropout(f))
        return (out, f) if return_feat else out

class ViTT(nn.Module):
    def __init__(self, nc, pretrained=True, mixstyle=False, tfs=False):
        super().__init__()
        self.tfs = tfs; self.tfs_d, self.tfs_alpha, self.tfs_first = 0.5, 0.1, 0.7   # TFS-ViT defaults: d, Beta alpha, first 70% of blocks
        import timm
        self.m = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, num_classes=nc)
        self.feat_dim = self.m.num_features
        self.ms = MixStyle(tokens=True) if mixstyle else None
        self.ms_after = {0, 1, 2}           # first transformer blocks
    def embed(self, x):
        m = self.m
        x = m.patch_embed(x); x = m._pos_embed(x); x = m.patch_drop(x); x = m.norm_pre(x)
        tfs_layer = np.random.randint(int(self.tfs_first * len(m.blocks))) if (self.tfs and self.training) else -1
        for i, blk in enumerate(m.blocks):
            x = blk(x)
            if i == tfs_layer: x = self._tfs(x)
            if self.ms is not None and i in self.ms_after:
                cls, tok = x[:, :1], x[:, 1:]
                x = torch.cat([cls, self.ms(tok)], 1)
        x = m.norm(x)
        return m.forward_head(x, pre_logits=True)
    def _tfs(self, x, eps=1e-6):
        """TFS-ViT (Noori et al., Pattern Recognition 2024), official TokenLevelFeatureStylization: mix token-wise
        mean/std between images in the batch (lambda ~ Beta(0.1, 0.1)) and replace a random d = 50% of the tokens
        of each image with their stylised version."""
        B, N, _ = x.shape
        mu = x.mean(1, keepdim=True).detach(); sig = (x.var(1, keepdim=True) + eps).sqrt().detach()
        lam = torch.distributions.Beta(self.tfs_alpha, self.tfs_alpha).sample((B, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        xs = (x - mu) / sig * (sig * lam + sig[perm] * (1 - lam)) + (mu * lam + mu[perm] * (1 - lam))
        sel = torch.rand(B, N, device=x.device).argsort(1) < int(self.tfs_d * N)
        return torch.where(sel.unsqueeze(-1), xs, x)
    def forward(self, x, return_feat=False):
        f = self.embed(x); out = self.m.head(self.m.head_drop(f))
        return (out, f) if return_feat else out

class ConvNeXtT(nn.Module):
    """ConvNeXt-T (torchvision IMAGENET1K_V1), the backbone of Farooq and Villing (2024)."""
    def __init__(self, nc, pretrained=True, mixstyle=False):
        super().__init__()
        from torchvision.models import convnext_tiny
        m = convnext_tiny(weights='IMAGENET1K_V1' if pretrained else None)
        self.features, self.avgpool = m.features, m.avgpool
        self.norm, self.flat = m.classifier[0], m.classifier[1]
        self.feat_dim = m.classifier[2].in_features
        self.fc = nn.Linear(self.feat_dim, nc)
    def embed(self, x):
        return self.flat(self.norm(self.avgpool(self.features(x))))
    def forward(self, x, return_feat=False):
        f = self.embed(x); out = self.fc(f)
        return (out, f) if return_feat else out

def build(backbone, nc, pretrained, mixstyle=False, tfs=False):
    if backbone == 'convnext':
        if mixstyle or tfs: raise SystemExit('Use --backbone convnext with --method erm only')
        return ConvNeXtT(nc, pretrained)
    if tfs:
        if backbone != 'vit': raise SystemExit('TFS-ViT is defined for ViT backbones only: use --backbone vit')
        return ViTT(nc, pretrained, tfs=True)
    return (EffNet if backbone == 'effnet' else ViTT)(nc, pretrained, mixstyle)

def make_opt(backbone, params):
    if backbone == 'effnet':                         # exp1-4 settings
        opt = torch.optim.Adam(params, lr=1e-3)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
    else:                                            # exp5-8 settings
        opt = torch.optim.AdamW(params, lr=1e-4); sch = None
    return opt, sch

# ----------------------------------------------------------------------------- MIRO
class MIRO(nn.Module):
    """MIRO (Cha et al., ECCV'22) on the final pooled features: frozen ImageNet oracle f0,
    mean encoder (identity-initialised linear) and per-dimension variance encoder.
    loss = CE + lambda * mean( log var + (f0 - mean(f))^2 / var )."""
    def __init__(self, model, oracle, lam=0.1):
        super().__init__()
        d = model.feat_dim
        self.model, self.oracle, self.lam = model, oracle, lam
        for p in self.oracle.parameters(): p.requires_grad_(False)
        self.mean_enc = nn.Linear(d, d); nn.init.eye_(self.mean_enc.weight); nn.init.zeros_(self.mean_enc.bias)
        self.var_param = nn.Parameter(torch.full((d,), 0.1))   # softplus(0.1) ~ 0.74 initial variance
    def loss(self, x, y):
        out, f = self.model(x, return_feat=True)
        self.oracle.eval()
        with torch.no_grad(): f0 = self.oracle.embed(x)
        var = F.softplus(self.var_param) + 1e-6
        reg = (torch.log(var) + (f0 - self.mean_enc(f)) ** 2 / var).mean()
        return F.cross_entropy(out, y) + self.lam * reg, out

# ----------------------------------------------------------------------------- CLIPood
class CLIPood(nn.Module):
    """CLIPood (Shu et al., ICML'23), simplified: fine-tune the CLIP image encoder against frozen class text
    embeddings with the margin metric softmax (MMS); Beta moving average (BMA) of weights in train_clipood()."""
    def __init__(self, classes, pretrained=True, lam=0.3, arch='ViT-B-16', tag='openai'):
        super().__init__()
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=tag if pretrained else None)
        tok = open_clip.get_tokenizer(arch)
        names = [c.replace('___', ' ').replace('_', ' ').lower() for c in classes]
        prompts = [f'a photo of a {n} leaf, a type of plant disease.' if 'healthy' not in n
                   else f'a photo of a healthy {n.replace(" healthy", "")} leaf, a type of plant.' for n in names]
        with torch.no_grad():
            t = model.encode_text(tok(prompts)); t = F.normalize(t, dim=-1)
        self.register_buffer('text', t)
        self.visual, self.lam = model.visual, lam
        self.logit_scale = model.logit_scale.exp().item()
        self.feat_dim = t.shape[1]
    def forward(self, x, y=None):
        f = F.normalize(self.visual(x), dim=-1)
        cos = f @ self.text.t()
        if y is None or not self.training:
            return self.logit_scale * cos
        # MMS: add a margin lam * (1 - <t_j, t_y>) to non-target classes, scaled by text similarity
        sim = self.text @ self.text.t()                      # (C, C)
        margin = self.lam * (1 - sim[y])                     # (B, C), zero for the target class
        return self.logit_scale * (cos + margin)



# ----------------------------------------------------------------------------- RandConv
def randconv(x, kernel_sizes=(1, 3, 5, 7)):
    """RandConv (Xu et al., ICLR'21), 'mix' variant: a random multi-scale convolution (kernel size drawn from
    {1, 3, 5, 7}, weights ~ N(0, 1/(3k^2))) applied to the input image and mixed with it, alpha ~ U(0, 1)."""
    k = random.choice(kernel_sizes)
    w = torch.randn(3, 3, k, k, device=x.device) / math.sqrt(3 * k * k)
    out = F.conv2d(x, w, padding=k // 2)
    a = random.random()
    return a * x + (1 - a) * out

# ----------------------------------------------------------------------------- SAGM
def _bn_momentum(model, backup=None):
    """disable (backup=None -> returns backup) or restore BN running-stat updates, as in the official SAGM code"""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if backup is None:
        b = [m.momentum for m in bns]
        for m in bns: m.momentum = 0
        return b
    for m, v in zip(bns, backup): m.momentum = v

def sagm_step(model, opt, x, y, rho=0.05, alpha=0.001, eps=1e-12):
    """One SAGM step (Wang et al., CVPR'23), following the official implementation:
    g at w; perturb w <- w + g * (rho/||g|| - alpha); g_p at the perturbed point (BN stats frozen);
    update direction = 0.5 * (g + g_p); restore w; base optimiser step."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y); loss.backward()
    with torch.no_grad():
        gn = torch.norm(torch.stack([p.grad.norm(2) for p in params if p.grad is not None]), 2)
        scale = rho / (gn + eps) - alpha
        old_g, e_w = {}, {}
        for p in params:
            if p.grad is None: continue
            old_g[p] = p.grad.detach().clone(); e_w[p] = p.grad * scale; p.add_(e_w[p])
    backup = _bn_momentum(model)
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    with torch.no_grad():
        for p in params:
            if p.grad is None or p not in old_g: continue
            p.grad.add_(old_g[p] * 0.5 - p.grad * 0.5)
            p.sub_(e_w[p])
    _bn_momentum(model, backup)
    opt.step()
    return loss.item()

# ----------------------------------------------------------------------------- RISE
RISE_NAMES = ['healthy apple leaf', 'apple leaf with rust', 'bell pepper leaf with bacterial spot', 'healthy bell pepper leaf',
              'grape leaf with black rot', 'healthy grape leaf', 'healthy tomato leaf', 'tomato leaf with septoria leaf spot']
RISE_TEMPLATES = ['a photo of a {}.', 'a bad photo of a {}.', 'a close-up photo of a {}.', 'a photo of a {} in a field.',
                  'a photo of a {} on a plant.', 'a cropped photo of a {}.', 'a bright photo of a {}.']
RISE_ANCHORS = ['bright photo', 'corrupted photo', 'dark photo', 'good photo']    # condition anchors (official code, single-domain datasets)

class RISE(nn.Module):
    """RISE (Huang et al., ICCV'23), following the official train_rise.py: a frozen CLIP ViT-B/16 teacher; the student
    projects its features to 512-d and its classifier is initialised with the CLIP text embeddings.
    loss = w_kd * T^2 * KL(student || teacher) + w_ce * CE + w_d * (absolute distance + relative distance)."""
    def __init__(self, student, T=3.0, w_kd=0.2, w_ce=0.8, w_d=1.0, pretrained=True):
        super().__init__()
        import open_clip
        clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai' if pretrained else None)
        tok = open_clip.get_tokenizer('ViT-B-16')
        clip.eval()
        for p in clip.parameters(): p.requires_grad_(False)
        with torch.no_grad():
            ems = torch.stack([clip.encode_text(tok([t.format(c) for c in RISE_NAMES])) for t in RISE_TEMPLATES]).mean(0)
            anchors = torch.stack([clip.encode_text(tok([f'a {a} of a {c}' for c in RISE_NAMES])) for a in RISE_ANCHORS], 1)
        self.register_buffer('text_raw', ems.float())                       # (C, 512) before normalisation
        self.register_buffer('text', F.normalize(ems.float(), dim=-1))
        self.register_buffer('anchors', anchors.float())                    # (C, A, 512)
        self.clip, self.student = clip, student
        self.proj = nn.Linear(student.feat_dim, 512)
        self.fc = nn.Linear(512, len(RISE_NAMES))
        with torch.no_grad(): self.fc.weight.copy_(self.text)
        self.T, self.w_kd, self.w_ce, self.w_d = T, w_kd, w_ce, w_d
    def train(self, mode=True):
        super().train(mode); self.clip.eval(); return self
    def forward(self, x, return_emb=False):
        emb = self.proj(self.student.embed(x)); out = self.fc(emb)
        return (emb, out) if return_emb else out
    def loss(self, x, y):
        with torch.no_grad():
            t_img = F.normalize(self.clip.encode_image(x).float(), dim=-1)
            t_logits = 100.0 * t_img @ self.text.t()
        emb, out = self(x, return_emb=True)
        T = self.T
        kd = F.kl_div(F.log_softmax(out / T, 1), F.softmax(t_logits / T, 1), reduction='batchmean') * T * T
        ce = F.cross_entropy(out, y)
        ad = F.cosine_embedding_loss(F.normalize(emb, dim=-1), self.text[y], torch.ones(len(y), device=x.device))
        anc = self.anchors[y]                                               # (B, A, 512)
        d_t = F.softmax(F.cosine_similarity(self.text_raw[y].unsqueeze(1), anc, dim=-1), 1)
        d_s = F.softmax(F.cosine_similarity(emb.unsqueeze(1), anc, dim=-1), 1)
        rd = F.mse_loss(d_s, d_t) * 10.0
        return self.w_kd * kd + self.w_ce * ce + self.w_d * (ad + rd), out


# ----------------------------------------------------------------------------- PEER
def _barlow(z1, z2, lam=0.005, eps=1e-5):
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + eps); z2 = (z2 - z2.mean(0)) / (z2.std(0) + eps)
    c = z1.t() @ z2 / z1.shape[0]
    on = (torch.diagonal(c) - 1).pow(2).sum()
    off = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()
    return on + lam * off

def _randaug_batch(x, mean, std, n_ops, mag):
    """RandAugment (torchvision) applied per image on de-normalised uint8 tensors, then re-normalised."""
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1); s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    u8 = ((x * s + m).clamp(0, 1) * 255).round().to(torch.uint8)
    ra = v2.RandAugment(num_ops=n_ops, magnitude=mag)
    out = torch.stack([ra(img) for img in u8.cpu()]).to(x.device).float() / 255
    return (out - m) / s

def train_peer(args, loaders, dev, nc):
    """PEER (Cho, Hwang and Lee, CVPR'25). Stage 1: the task model F is the standard source model (our ERM pipeline).
    Stage 2: a proxy P (initialised from F) is trained on x and a RandAugment view x_bar with
    CE(x) + CE(x_bar) + w * BarlowTwins(R(H_F(x)), R(H_P(x_bar))); F receives no gradients. Every k epochs a snapshot
    of P is stored, F is set to the uniform average of all snapshots (initial F included) and the RandAugment policy
    is resampled. The final F is evaluated, with no early stopping, as in the paper."""
    tr = loaders[0]
    F_model = train_standard(args, loaders, dev, nc)               # stage 1
    print('  PEER stage 2', flush=True)
    P = copy.deepcopy(F_model).train()
    d = F_model.feat_dim
    R = nn.Sequential(nn.Linear(d, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
                      nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Linear(1024, 1024)).to(dev)
    opt = torch.optim.Adam(list(P.parameters()) + list(R.parameters()), lr=args.peer_lr)
    avg = {k: v.detach().clone().float() for k, v in F_model.state_dict().items()}; n = 1
    for p in F_model.parameters(): p.requires_grad_(False)
    n_ops, mag = random.randint(1, 4), random.randint(5, 15)
    for ep in range(args.peer_epochs):
        F_model.eval()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            xb = _randaug_batch(x, MEAN, STD, n_ops, mag)
            with torch.no_grad(): hf = F_model.embed(x)
            out_x = P(x); out_b, hp_b = P(xb, return_feat=True)
            loss = (F.cross_entropy(out_x, y) + F.cross_entropy(out_b, y)
                    + args.peer_w * _barlow(R(hf), R(hp_b)))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        print(f'  peer epoch {ep}: loss {loss.item():.4f}', flush=True)
        if (ep + 1) % args.peer_k == 0 or ep == args.peer_epochs - 1:
            sd = P.state_dict(); n += 1
            for k in avg:
                if sd[k].dtype.is_floating_point: avg[k] += (sd[k].detach().float() - avg[k]) / n
                else: avg[k] = sd[k].detach().clone()
            F_model.load_state_dict({k: v.to(F_model.state_dict()[k].dtype) for k, v in avg.items()})
            n_ops, mag = random.randint(1, 4), random.randint(5, 15)
    return F_model


# ----------------------------------------------------------------------------- ViT-MoE (Salman et al., 2025)
class ViTMoE(nn.Module):
    """ViT + mixture of experts (Salman, Muhammad and Han, Frontiers in Plant Science 16, 1522985, 2025), following the
    official run.py/model.py (github.com/salman32140/Vit_MoE): google/vit-base-patch16-224-in21k, LayerNorm on the
    pooler output, a gating MLP (768 -> 384 -> E, softmax) and E expert MLPs (768 -> 512 -> C, GELU, dropout 0.1,
    softmax); output = gate-weighted sum of the top-k experts (k = E, as in run.py)."""
    def __init__(self, nc, n_experts=4, pretrained=True):
        super().__init__()
        from transformers import ViTModel, ViTConfig
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k') if pretrained else ViTModel(ViTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128))
        d = self.vit.config.hidden_size
        self.ln = nn.LayerNorm(d)
        self.gate = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d // 2, n_experts))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, nc)) for _ in range(n_experts)])
        self.k = n_experts
    def forward_all(self, x):
        h = self.ln(self.vit(pixel_values=x).pooler_output)
        g = torch.softmax(self.gate(h), 1)
        e = torch.stack([torch.softmax(ex(h), 1) for ex in self.experts], 1)        # (B, E, C)
        v, i = torch.topk(g, self.k, 1); v = v / v.sum(1, keepdim=True)
        out = (v.unsqueeze(-1) * torch.gather(e, 1, i.unsqueeze(-1).expand(-1, -1, e.size(-1)))).sum(1)
        return out, g, e
    def forward(self, x):
        return torch.log(self.forward_all(x)[0] + 1e-8)     # log-probabilities, so evaluate() and argmax are unchanged

def train_vitmoe(args, loaders, dev, nc):
    """Official optimisation: Adam for experts (1e-3), gating (8.25e-5) and the ViT encoder (5.58e-6; patch embedding
    frozen); loss = CE(output, y) + CE(gate, best expert for y), CE applied to the softmax outputs as in run.py;
    45 epochs; the epoch with the best validation accuracy is kept. No augmentation, inputs normalised with mean = std = 0.5."""
    tr, va = loaders[0], loaders[1]
    model = ViTMoE(nc, args.moe_experts, pretrained=not args.no_pretrained).to(dev)
    for p in model.vit.parameters(): p.requires_grad_(False)
    enc = getattr(model.vit, 'encoder', None) or model.vit.layers        # transformers 4: .encoder, 5: .layers
    for p in enc.parameters(): p.requires_grad_(True)
    opts = [torch.optim.Adam(model.experts.parameters(), lr=1e-3), torch.optim.Adam(model.gate.parameters(), lr=8.24654e-05),
            torch.optim.Adam(enc.parameters(), lr=5.5839e-06)]
    best, best_state = None, None
    for ep in range(args.moe_epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            out, g, e = model.forward_all(x)
            tgt_gate = e[torch.arange(len(e)), :, y].argmax(1)
            loss = F.cross_entropy(g, tgt_gate) + F.cross_entropy(out, y)
            for o in opts: o.zero_grad(set_to_none=True)
            loss.backward()
            for o in opts: o.step()
        v = evaluate(model, va, dev)
        print(f'  epoch {ep}: val acc {v["acc"]:.4f}', flush=True)
        if best is None or v['acc'] > best:
            best, best_state = v['acc'], {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model

# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); ys, ps, loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x); loss += F.cross_entropy(out, y, reduction='sum').item(); n += y.numel()
        ps.append(out.argmax(1).cpu()); ys.append(y.cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return dict(loss=loss / n, acc=float((y == p).mean()), f1=float(f1_score(y, p, average='macro')),
                cm=confusion_matrix(y, p).tolist())

# ----------------------------------------------------------------------------- training
def train_standard(args, loaders, dev, nc):
    tr, va, _, _, _ = loaders
    model = build(args.backbone, nc, not args.no_pretrained, mixstyle=(args.method == 'mixstyle'), tfs=(args.method == 'tfsvit')).to(dev)
    miro = None
    if args.method == 'rise':
        rise = RISE(model, pretrained=not args.no_pretrained).to(dev)
        params = list(model.parameters()) + list(rise.proj.parameters()) + list(rise.fc.parameters())
        opt, sch = make_opt(args.backbone, params)
        return _fit(args, rise, rise.loss, opt, sch, loaders, dev)
    if args.method == 'miro':
        oracle = build(args.backbone, nc, not args.no_pretrained).to(dev)
        miro = MIRO(model, oracle, args.miro_lambda).to(dev)
        params = [p for n, p in miro.named_parameters() if not n.startswith('oracle.')]
    else:
        params = model.parameters()
    opt, sch = make_opt(args.backbone, params)
    if args.method == 'sagm':
        return _fit(args, model, None, opt, sch, loaders, dev, sagm=True)
    return _fit(args, model, (miro.loss if miro is not None else None), opt, sch, loaders, dev)

def _fit(args, model, loss_fn, opt, sch, loaders, dev, sagm=False):
    tr, va = loaders[0], loaders[1]
    best, best_state, wait = None, None, 0
    # early stopping exactly as in the notebooks: val loss (EffNet) / val accuracy (ViT), patience 5, restore best
    mon_loss = args.backbone == 'effnet'
    for ep in range(args.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            if sagm:
                sagm_step(model, opt, x, y, rho=args.sagm_rho, alpha=args.sagm_alpha); continue
            if args.method == 'randconv': x = randconv(x)
            loss = loss_fn(x, y)[0] if loss_fn is not None else F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if sch: sch.step()
        v = evaluate(model, va, dev)
        score = -v['loss'] if mon_loss else v['acc']
        print(f'  epoch {ep}: val loss {v["loss"]:.4f} acc {v["acc"]:.4f}', flush=True)
        if best is None or score > best:
            best, best_state, wait = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith('clip.')}, 0
        else:
            wait += 1
            if wait >= args.patience: print('  early stopping'); break
    model.load_state_dict(best_state, strict=False)
    return model

def train_swad(args, loaders, dev, nc):
    """SWAD (Cha et al., NeurIPS'21) with validation evaluated once per epoch.
    Dense (every-iteration) weight averaging; averaging window [ts, te] chosen from the source validation loss:
    ts = first epoch after which the loss stops decreasing for Ns epochs; te = first epoch after ts at which the
    loss exceeds r * min loss for Ne epochs (or the last epoch). BN statistics are recomputed for the average."""
    tr, va, _, _, _ = loaders
    model = build(args.backbone, nc, not args.no_pretrained).to(dev)
    opt, sch = make_opt(args.backbone, model.parameters())
    Ns, Ne, r = args.swad_ns, args.swad_ne, args.swad_r
    epoch_avgs, val_losses = [], []
    for ep in range(args.epochs):                      # SWAD trains for the full schedule (no early stopping)
        model.train(); acc_sd, k = None, 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            sd = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}
            if acc_sd is None: acc_sd = sd
            else:
                for n in acc_sd: acc_sd[n] += sd[n]
            k += 1
        if sch: sch.step()
        epoch_avgs.append({n: v / k for n, v in acc_sd.items()})
        v = evaluate(model, va, dev); val_losses.append(v['loss'])
        print(f'  epoch {ep}: val loss {v["loss"]:.4f} acc {v["acc"]:.4f}', flush=True)
    L = val_losses; T = len(L)
    ts = next((i for i in range(T) if all(L[j] >= L[i] for j in range(i + 1, min(T, i + 1 + Ns)))), 0)
    lmin = min(L[ts:]); te = T - 1
    for i in range(ts + 1, T):
        if all(L[j] > r * lmin for j in range(i, min(T, i + Ne))) and i + Ne <= T:
            te = i - 1; break
    avg = {n: sum(epoch_avgs[e][n] for e in range(ts, te + 1)) / (te - ts + 1) for n in epoch_avgs[0]}
    model.load_state_dict({**model.state_dict(), **{n: v.to(dev) for n, v in avg.items()}}, strict=False)
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        model.train()
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm): m.reset_running_stats(); m.momentum = None
            for x, _ in tr: model(x.to(dev))
    print(f'  SWAD window: epochs {ts} to {te}')
    model.swad_window = (ts, te)
    return model

def train_clipood(args, loaders, dev, nc, classes):
    tr, va, _, _, _ = loaders
    model = CLIPood(classes, pretrained=not args.no_pretrained, lam=args.clipood_lambda).to(dev)
    opt = torch.optim.AdamW(model.visual.parameters(), lr=args.clipood_lr, weight_decay=0.1)
    steps = args.clipood_epochs * len(tr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    # BMA: average weights over iterations with Beta(0.5, 0.5) density weights over normalised training time
    beta = torch.distributions.Beta(0.5, 0.5)
    bma, wsum, t = None, 0.0, 0
    for ep in range(args.clipood_epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            loss = F.cross_entropy(model(x, y), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); t += 1
            u = min(max(t / steps, 1e-3), 1 - 1e-3)
            w = beta.log_prob(torch.tensor(u)).exp().item()
            sd = {n: p.detach().float().cpu() for n, p in model.visual.named_parameters()}
            if bma is None: bma = {n: v.clone() * w for n, v in sd.items()}
            else:
                for n in bma: bma[n] += sd[n] * w
            wsum += w
        v = evaluate(model, va, dev); print(f'  epoch {ep}: val loss {v["loss"]:.4f} acc {v["acc"]:.4f}', flush=True)
    model.visual.load_state_dict({n: (bma[n] / wsum).to(dev) for n in bma}, strict=False)
    return model

# ----------------------------------------------------------------------------- main
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def run(args, seed):
    set_seed(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip = args.method in ('clipood', 'clipzs')
    loaders = make_loaders(args.pv, args.pd, args.batch_size, args.workers, clip=clip, augmix=(args.method == 'augmix'),
                           syn_dir=(args.syn_dir if args.method == 'dafusion' else None), plain=(args.method == 'vitmoe'))
    classes = loaders[4]; nc = len(classes)
    t0 = time.time()
    if args.method == 'swad': model = train_swad(args, loaders, dev, nc)
    elif args.method == 'clipzs': model = CLIPood(classes, pretrained=not args.no_pretrained).to(dev).eval()   # zero-shot reference, no training
    elif clip: model = train_clipood(args, loaders, dev, nc, classes)
    elif args.method == 'peer': model = train_peer(args, loaders, dev, nc)
    elif args.method == 'vitmoe': model = train_vitmoe(args, loaders, dev, nc)
    else: model = train_standard(args, loaders, dev, nc)
    src = evaluate(model, loaders[2], dev); tgt = evaluate(model, loaders[3], dev)
    row = dict(method=args.method + (f'+{args.tag}' if args.tag else ''), backbone='clip-vitb16' if clip else ('vit-base-in21k' if args.method == 'vitmoe' else args.backbone), seed=seed,
               source_acc=round(src['acc'], 4), target_acc=round(tgt['acc'], 4), target_f1=round(tgt['f1'], 4),
               ds=round(src['acc'] - tgt['acc'], 4), minutes=round((time.time() - t0) / 60, 1),
               extra=json.dumps(dict(swad_window=getattr(model, 'swad_window', None), miro_lambda=args.miro_lambda if args.method == 'miro' else None)))
    print(row, flush=True)
    new = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
    with open(args.out.replace('.csv', f'_cm_{args.method}_{row["backbone"]}_s{seed}.json'), 'w') as f:
        json.dump(dict(classes=classes, target_cm=tgt['cm']), f)

def summarise(path):
    import pandas as pd
    df = pd.read_csv(path)
    g = df.groupby(['method', 'backbone'])[['source_acc', 'target_acc', 'target_f1', 'ds']].agg(['mean', 'std', 'count'])
    pd.set_option('display.width', 200); print(g.round(4))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pv', nargs='+', default=['C:/Adil PhD Work/WP2 CCLR_ECCLR/data/plantvillage'])
    ap.add_argument('--pd', nargs='+', default=['C:/Adil PhD Work/WP2 CCLR_ECCLR/data/plantdoc'])
    ap.add_argument('--method', choices=['erm', 'mixstyle', 'swad', 'miro', 'clipood', 'sagm', 'rise', 'randconv', 'augmix', 'peer', 'tfsvit', 'clipzs', 'dafusion', 'vitmoe'])
    ap.add_argument('--backbone', choices=['effnet', 'vit', 'convnext'], default='effnet')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=25); ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=32); ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--miro-lambda', type=float, default=0.1)
    ap.add_argument('--syn-dir', default='C:/Adil PhD Work/WP2 CCLR_ECCLR/data/plantvillage_dafusion', help='output of gen_dafusion.py')
    ap.add_argument('--moe-experts', type=int, default=4); ap.add_argument('--moe-epochs', type=int, default=45)
    ap.add_argument('--peer-epochs', type=int, default=25); ap.add_argument('--peer-k', type=int, default=5)
    ap.add_argument('--peer-w', type=float, default=2.0); ap.add_argument('--peer-lr', type=float, default=1e-4)
    ap.add_argument('--sagm-rho', type=float, default=0.05); ap.add_argument('--sagm-alpha', type=float, default=0.001)
    ap.add_argument('--tag', default='', help='label written to the results, e.g. pv, cclr, ecclr')
    ap.add_argument('--swad-ns', type=int, default=3); ap.add_argument('--swad-ne', type=int, default=3); ap.add_argument('--swad-r', type=float, default=1.2)
    ap.add_argument('--clipood-epochs', type=int, default=10); ap.add_argument('--clipood-lr', type=float, default=1e-5)
    ap.add_argument('--clipood-lambda', type=float, default=0.3)
    ap.add_argument('--no-pretrained', action='store_true', help='smoke test only')
    ap.add_argument('--out', default='dg_results.csv')
    ap.add_argument('--summarise', action='store_true')
    a = ap.parse_args()
    if a.summarise: summarise(a.out)
    else:
        for s in a.seeds: run(a, s)
