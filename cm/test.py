# save this as `test_unet_conditioned.py` next to your UNet file
import argparse
import torch
import torch.nn as nn

# adjust this import to match your package layout
# e.g. if your UNet file is `models/unet_conditioned.py`, use:
# from models.unet_conditioned import UNetModel
from cm.unet_radio import UNetModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--in_channels', type=int, default=1)
    p.add_argument('--out_channels', type=int, default=1)
    p.add_argument('--model_channels', type=int, default=64)
    p.add_argument('--num_res_blocks', type=int, default=2)
    p.add_argument('--attention', type=str, default='4,8,16', help='comma-separated ds factors')
    p.add_argument('--enable_cond', action='store_true')
    p.add_argument('--cond_net', type=str, default='resnet50', choices=['resnet50','vgg16','efficientnet_b7','swin_b'])
    p.add_argument('--dual_head', action='store_true')
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--consistency', action='store_true', help='if set, feeds sigma in [0,1] and expects two outputs')
    p.add_argument('--fp16', action='store_true')
    args = p.parse_args()

    attn_res = {int(x) for x in args.attention.split(',') if x}

    model = UNetModel(
        image_size=args.image_size,
        in_channels=args.in_channels,
        model_channels=args.model_channels,
        out_channels=args.out_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=attn_res,
        enable_cond=args.enable_cond,
        cond_net=args.cond_net,
        cond_pretrained=False,  # set True if you want actual pretrained weights
        dual_head=args.dual_head,
    ).to(args.device)

    model.train()

    B = args.batch
    H = W = args.image_size
    x = torch.randn(B, args.in_channels, H, W, device=args.device)

    # cond image must be 3-channel for torchvision backbones; resize freely
    cond = None
    if args.enable_cond:
        cond = torch.randn(B, 3, H, W, device=args.device)

    if args.consistency:
        sigma = torch.rand(B, device=args.device)  # values in [0,1]
    else:
        # pretend timesteps for standard diffusion (ints or floats). Using floats here.
        sigma = torch.randint(0, 1000, (B,), device=args.device).float()

    # forward (with optional autocast)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and (args.device.startswith('cuda')))
    criterion = nn.MSELoss()

    if args.fp16 and args.device.startswith('cuda'):
        with torch.cuda.amp.autocast():
            out = model(x, sigma, cond=cond, consistency=args.consistency)
            if args.consistency and args.dual_head:
                y1, y2 = out
                loss = criterion(y1, torch.zeros_like(y1)) + criterion(y2, torch.zeros_like(y2))
            else:
                y = out[0] if isinstance(out, (tuple, list)) else out
                loss = criterion(y, torch.zeros_like(y))
    else:
        out = model(x, sigma, cond=cond, consistency=args.consistency)
        if args.consistency and args.dual_head:
            y1, y2 = out
            loss = criterion(y1, torch.zeros_like(y1)) + criterion(y2, torch.zeros_like(y2))
        else:
            y = out[0] if isinstance(out, (tuple, list)) else out
            loss = criterion(y, torch.zeros_like(y))

    # backward to ensure graph is valid
    scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()

    # print shapes and a small checksum
    def tshape(t):
        return list(t.shape)

    if args.consistency and args.dual_head:
        print('OK: dual-head consistency forward/backward succeeded.')
        print('pred1 shape:', tshape(y1), ' pred2 shape:', tshape(y2))
        print('loss:', float(loss.detach().cpu()))
    else:
        y = out[0] if isinstance(out, (tuple, list)) else out
        print('OK: single-head forward/backward succeeded.')
        print('out shape:', tshape(y))
        print('loss:', float(loss.detach().cpu()))

if __name__ == '__main__':
    main()
