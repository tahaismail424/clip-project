"""Gradient check for global_anchor_step.

Compares p.grad produced by global_anchor_step (cached embeddings, blocked
anchor loss, minibatch re-backward) against a single plain full-batch
CLIP backward on the same N samples.  Run on a GPU node from csce689_iSogCLR/.
"""
import sys, os, json
from types import SimpleNamespace

REPO = '/scratch/user/tahaismail424/coursework/cs633/clip-project/csce689_iSogCLR'
sys.path.insert(0, os.path.join(REPO, 'bimodal_exps'))

import torch
from torch.utils.data import Subset
import clip as C
from models.model_clip import CLIP
from models.losses import sampled_anchor_clip_loss
from dataset import create_train_dataset, create_train_loader
from transformers import AutoTokenizer

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

N, MB, BLOCK, TEMP, REG = 100, 16, 32, 0.01, 1e-5
device = torch.device('cuda')
data = os.path.join(REPO, '../data')
args = SimpleNamespace(train_file=os.path.join(data, 'clip_train/cc3m_train_subset.json'),
                       train_image_root=os.path.join(data, 'cc3m_subset_100k'), image_res=256)
ds = Subset(create_train_dataset('re', args, deterministic=True), list(range(N)))
loader = create_train_loader(ds, None, MB, 4, None, global_anchor=True)
tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')

torch.manual_seed(0)
model = CLIP(image_encoder='resnet50', text_encoder='distilbert-base-uncased', embed_dim=256,
             init_model=True, world_size=1, ita_type='clip', temp=TEMP, bsz=MB, distributed=False).to(device)
named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
params = [p for _, p in named]
names = [n for n, _ in named]


def reference_grads(with_l2, per_minibatch=False):
    """Plain autograd, loss over all N anchors, single backward.
    per_minibatch=False: all N samples in ONE encoder forward (different conv algos than GA).
    per_minibatch=True : same minibatches as GA, graph kept, one backward -> same arithmetic as GA."""
    model.eval()
    model.zero_grad(set_to_none=True)
    if per_minibatch:
        fi, ft = [], []
        for image, text, _, _ in loader:
            ti = tok(list(text), padding='max_length', truncation=True, max_length=30, return_tensors='pt').to(device)
            fi.append(model.encode_image(image.to(device))); ft.append(model.encode_text(ti))
        img_f, txt_f = torch.cat(fi), torch.cat(ft)
    else:
        imgs, txts = [], []
        for image, text, _, _ in loader:
            imgs.append(image); txts += list(text)
        image = torch.cat(imgs).to(device)
        text_input = tok(txts, padding='max_length', truncation=True, max_length=30, return_tensors='pt').to(device)
        img_f, txt_f = model.encode_image(image), model.encode_text(text_input)
    loss = sampled_anchor_clip_loss(img_f, txt_f, torch.arange(N, device=device), TEMP)
    if with_l2:
        loss = loss + 0.5 * REG * sum(p.pow(2).sum() for p in params if p.ndim > 1)
    loss.backward()
    return loss.item(), [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]


def ga_grads(optimizer):
    stats = {}
    loss = C.global_anchor_step(model, loader, optimizer, tok, 0, device, TEMP,
                                block_size=BLOCK, reg_strength=REG, stats=stats)
    return loss.item(), [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params], stats


def compare(name, ref, got):
    (l_ref, g_ref), (l_got, g_got) = ref, got
    num = sum((a - b).pow(2).sum() for a, b in zip(g_got, g_ref)).sqrt().item()
    den = sum(b.pow(2).sum() for b in g_ref).sqrt().item()
    per = sorted((((a - b).norm() / (b.norm() + 1e-12)).item(), b.norm().item(), n)
                 for a, b, n in zip(g_got, g_ref, names) if b.norm() > 1e-8)
    worst = per[-1][0]
    print('  worst params (rel_err, |grad|, name):', [(round(r,4), round(g,6), n) for r,g,n in per[-4:]], flush=True)
    res = dict(case=name, loss_ref=l_ref, loss_ga=l_got, loss_abs_diff=abs(l_ref - l_got),
               grad_rel_err_global=num / den, grad_rel_err_worst_param=worst, ref_grad_norm=den)
    res['PASS'] = res['grad_rel_err_global'] < 1e-4 and res['loss_abs_diff'] < 1e-4
    print(json.dumps(res), flush=True)
    return res['PASS']


ok = True
# Noise floor: one-big-batch reference vs same-minibatch reference (no GradCache involved at all)
compare('NOISE_FLOOR ref_fullbatch vs ref_minibatch', reference_grads(False), reference_grads(False, per_minibatch=True))
# AdamW-style path: no L2 term inside global_anchor_step
adam = torch.optim.AdamW(params, lr=0.0)
l, g, st = ga_grads(adam)
compare('adamw_path vs ref_fullbatch', reference_grads(False), (l, g))
ok &= compare('adamw_path vs ref_minibatch (exact test)', reference_grads(False, per_minibatch=True), (l, g))
print('stats:', json.dumps(st), flush=True)

# LBFGS path: L2 term added inside global_anchor_step
lbfgs = torch.optim.LBFGS(params, lr=1)
l, g, st = ga_grads(lbfgs)
ok &= compare('lbfgs_path vs ref_minibatch (exact test)', reference_grads(True, per_minibatch=True), (l, g))
print('stats:', json.dumps(st), flush=True)
print('ALL PASS' if ok else 'SOME CHECK FAILED', flush=True)
