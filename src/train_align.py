# SiT 학습 스크립트 (단일 GPU) + 선택적 REPA식 정렬 loss + held-out 평가
import os, csv, argparse
from time import time
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from diffusers.models import AutoencoderKL

from models import SiT_models
from transport import create_transport
from train_utils import parse_transport_args


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    cy = (arr.shape[0] - image_size) // 2
    cx = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[cy:cy + image_size, cx:cx + image_size])


def main(args):
    assert torch.cuda.is_available(), "GPU required"
    dist.init_process_group("gloo")          # single-GPU Colab
    rank, world = dist.get_rank(), dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.manual_seed(args.global_seed * world + rank)
    torch.cuda.set_device(device)
    local_bs = args.global_batch_size // world

    exp_name = args.exp_name or f"{'align' if args.align else 'base'}-seed{args.global_seed}"
    exp_dir = os.path.join(args.results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    csv_f = open(os.path.join(exp_dir, "metrics.csv"), "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["step", "l_simple", "l_align", "total"])
    eval_f = open(os.path.join(exp_dir, "eval.csv"), "w", newline="")
    eval_writer = csv.writer(eval_f)
    eval_writer.writerow(["step", "train_fixed", "heldout"])

    # ---- model ----
    latent_size = args.image_size // 8
    model = SiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    hidden = model.pos_embed.shape[-1]
    n_tokens = model.pos_embed.shape[1]

    feats = {}
    projector, encoder = None, None
    params = list(model.parameters())
    if args.align:
        from transformers import Dinov2Model
        encoder = Dinov2Model.from_pretrained(args.dino).to(device).eval()
        for p in encoder.parameters():
            p.requires_grad = False
        z_dim = encoder.config.hidden_size
        # projector 초기화가 전역 RNG를 소비하지 않도록 (baseline과 노이즈 샘플링 동일하게 유지)
        with torch.random.fork_rng(devices=[]):
            projector = nn.Sequential(
                nn.Linear(hidden, args.proj_hidden), nn.SiLU(),
                nn.Linear(args.proj_hidden, args.proj_hidden), nn.SiLU(),
                nn.Linear(args.proj_hidden, z_dim),
            )
        projector = projector.to(device)
        params += list(projector.parameters())
        model.blocks[args.encoder_depth - 1].register_forward_hook(
            lambda m, i, o: feats.__setitem__("h", o))
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    model = DDP(model, device_ids=[device])
    transport = create_transport(args.path_type, args.prediction, args.loss_weight,
                                 args.train_eps, args.sample_eps)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0)

    # ---- data ----
    transform = transforms.Compose([
        transforms.Lambda(lambda im: center_crop_arr(im, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True),
    ])
    eval_transform = transforms.Compose([       # 평가용: flip 없음
        transforms.Lambda(lambda im: center_crop_arr(im, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=local_bs, shuffle=False, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(f"[{exp_name}] {args.model}, align={args.align}, train_images={len(dataset)}, "
          f"tokens={n_tokens}, seed={args.global_seed}", flush=True)

    # ---- 고정 평가셋: latent를 한 번만 만들어 재사용 (학습 난수 순서에 영향 없도록 RNG 격리) ----
    @torch.no_grad()
    def build_eval_set(folder):
        ds = ImageFolder(folder, transform=eval_transform)
        dl = DataLoader(ds, batch_size=20, shuffle=False, num_workers=0)
        xs, ys = [], []
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(args.eval_seed)
            for img, y in dl:
                xs.append(vae.encode(img.to(device)).latent_dist.sample().mul_(0.18215))
                ys.append(y.to(device))
        return torch.cat(xs), torch.cat(ys)

    train_x, train_y = build_eval_set(args.data_path)     # 학습에 쓴 40장 (고정 노이즈로 재측정)
    held_x, held_y = build_eval_set(args.eval_path)       # 학습에 쓰지 않은 40장

    @torch.no_grad()
    def evaluate(ex, ey):
        """고정 노이즈/t 로 L_simple 을 측정. 전역 RNG 상태는 복원된다."""
        model.eval()
        losses = []
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(args.eval_seed)
            for _ in range(args.eval_repeats):
                for i in range(0, ex.shape[0], args.eval_batch):
                    xb, yb = ex[i:i + args.eval_batch], ey[i:i + args.eval_batch]
                    losses.append(transport.training_losses(model, xb, dict(y=yb))["loss"].mean().item())
        model.train()
        return float(np.mean(losses))

    def run_eval(step):
        tr, ho = evaluate(train_x, train_y), evaluate(held_x, held_y)
        eval_writer.writerow([step, tr, ho])
        eval_f.flush()
        print(f"[{exp_name}] step={step:05d} EVAL train_fixed={tr:.4f} heldout={ho:.4f}", flush=True)

    @torch.no_grad()
    def dino_patch_feats(img):
        x = (img + 1) / 2
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False)
        x = (x - mean) / std
        return encoder(pixel_values=x).last_hidden_state[:, 1:]   # patch tokens (B, 256, z_dim)

    model.train()
    run_eval(0)
    step, t0 = 0, time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for img, y in loader:
            img, y = img.to(device), y.to(device)
            with torch.no_grad():
                x = vae.encode(img).latent_dist.sample().mul_(0.18215)
            l_simple = transport.training_losses(model, x, dict(y=y))["loss"].mean()

            if args.align:
                z = dino_patch_feats(img)
                z_tilde = projector(feats["h"])
                assert z_tilde.shape[1] == z.shape[1], \
                    f"token mismatch: SiT {z_tilde.shape[1]} vs DINOv2 {z.shape[1]} (SiT-S/2 필요)"
                l_align = -F.cosine_similarity(z_tilde, z, dim=-1).mean()
                loss = l_simple + args.align_lambda * l_align
            else:
                l_align = torch.zeros((), device=device)
                loss = l_simple

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            writer.writerow([step, l_simple.item(), l_align.item(), loss.item()])
            if step % args.log_every == 0:
                csv_f.flush()
                print(f"[{exp_name}] step={step:05d} L_simple={l_simple.item():.4f} "
                      f"L_align={l_align.item():.4f} ({step / (time() - t0):.2f} it/s)", flush=True)
            if step % args.eval_every == 0:
                run_eval(step)

    csv_f.close()
    eval_f.close()
    print(f"[{exp_name}] Done!", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--eval-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results_cmp")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-S/2")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=50)
    # 평가
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--eval-repeats", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=20)
    # 정렬 (REPA-style)
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--align-lambda", type=float, default=0.5)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--proj-hidden", type=int, default=2048)
    parser.add_argument("--dino", type=str, default="facebook/dinov2-small")
    parse_transport_args(parser)
    main(parser.parse_args())
