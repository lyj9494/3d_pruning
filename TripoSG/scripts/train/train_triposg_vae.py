import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import numpy as np

from TripoSG.triposg.models.autoencoders.autoencoder_kl_triposg import AutoencoderKLTripoSG
from scripts.train.utils.data_utils import SDFDataset
from scripts.utils.vae_utils import (
    sdf_loss,
    normal_loss,
    eikonal_loss,
)

def main(args):
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("train_vae")

    # Set seed
    set_seed(args.seed)

    # Load dataset
    train_dataset = SDFDataset(
        args.dataset_path,
        num_points=args.num_points,
        num_surface_points=args.num_surface_points,
        load_ram=args.load_ram,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # Load model
    model = AutoencoderKLTripoSG(
        in_channels=1,
        out_channels=1,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        latent_channels=256,
        norm_num_groups=32,
        in_res=args.resolution,
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Prepare everything with accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            points = batch["points"]
            sdf_gt = batch["sdf"]
            normals_gt = batch["normals"]
            surface_points = batch["surface_points"]

            with accelerator.accumulate(model):
                # Forward pass
                posterior = model.encode(sdf_gt.unsqueeze(1)).latent_dist
                z = posterior.sample()

                # Reshape predictions to match input points
                # Assuming sdf_pred_all is a grid, we need to sample from it
                # For simplicity, let's assume the decoder can take points as input
                # This part needs to be adapted based on the VAE's decoder implementation
                
                # For gradient calculation, we need to evaluate the decoder at specific points
                surface_points.requires_grad_()
                sdf_on_surface = model.decode(z, surface_points) # This is a placeholder, decoder needs to support this

                # Calculate gradients for normal and eikonal loss
                gradient = torch.autograd.grad(
                    outputs=sdf_on_surface,
                    inputs=surface_points,
                    grad_outputs=torch.ones_like(sdf_on_surface),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                
                # For SDF loss, we need predictions at the same points as the GT
                # This is a placeholder, assuming we can get predictions for the points
                sdf_pred = model.decode(z, points) 

                # Loss calculation
                loss_sdf = sdf_loss(sdf_pred, sdf_gt)
                loss_sn = normal_loss(gradient, normals_gt)
                loss_eik = eikonal_loss(gradient)
                loss_kl = posterior.kl().mean()

                total_loss = (
                    loss_sdf
                    + args.lambda_sn * loss_sn
                    + args.lambda_eik * loss_eik
                    + args.lambda_kl * loss_kl
                )

                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": total_loss.detach().item(),
                "loss_sdf": loss_sdf.detach().item(),
                "loss_sn": loss_sn.detach().item(),
                "loss_eik": loss_eik.detach().item(),
                "loss_kl": loss_kl.detach().item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                save_path = os.path.join(args.output_dir, f"vae-epoch-{epoch}.pt")
                torch.save(accelerator.unwrap_model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_vae")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--num_points", type=int, default=10000)
    parser.add_argument("--num_surface_points", type=int, default=2048)
    parser.add_argument("--load_ram", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--lambda_sn", type=float, default=0.1, help="Weight for surface normal loss")
    parser.add_argument("--lambda_eik", type=float, default=0.1, help="Weight for eikonal loss")
    parser.add_argument("--lambda_kl", type=float, default=1e-6, help="Weight for KL divergence")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
