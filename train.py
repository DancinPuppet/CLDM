import os
import torch
import time
import torch.nn.functional as F
from tqdm import tqdm
from eval import evaluate_model_with_metrics


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {checkpoint['epoch']}")
    return model, optimizer, scheduler, start_epoch


def train_CLDM(args, dataloader, dataloader_temp, model, optimizer, scheduler, device):
    print('model loaded!')
    start_epoch = 1
    early_stop_patience = 100 
    no_improve_epochs = 0 
    best_val_f1 = 0.0
    best_model_path = os.path.join(args.save_path, f"{args.model_name}_{args.graph_type}_best_f1.pt")
    log_file = os.path.join(args.output_save_path, f"{args.model_name}_{args.graph_type}_train_log.txt")
    if args.resume_train:
        model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint_path, device)
    for epoch in range(start_epoch, args.epochs+1):
        model.train()
        start_time = time.time()
        loss_sum = 0
        num_batches = 0
        print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        with tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as tbar:
            for idx, batch in enumerate(tbar):
                optimizer.zero_grad()
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                y = batch.y.to(device)
                batch_data = batch.batch
                loss_dict = model(x, edge_index, y, batch_data, epoch)
                loss = loss_dict['total_loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_sum += loss.item()
                num_batches += 1
                tbar.set_postfix({
                    'Total': f'{loss:.4f}',
                    'cross_loss': f'{loss_dict["cross_loss"]:.4f}',
                    'pred_loss': f'{loss_dict["pred_loss"]:.4f}',
                    'recon_loss': f'{loss_dict["recon_loss"]:.4f}',
                    'kl_loss': f'{loss_dict["kl_loss"]:.4f}'
                })

        avg_loss = loss_sum / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} finished, avg Loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")
        scheduler.step()
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} | AvgLoss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_path, f'{args.model_name}_{args.graph_type}_checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
        if epoch % args.epoch_test == 0:
            model.eval()
            with torch.no_grad():
                result = evaluate_model_with_metrics(model, dataloader_temp, device)
                current_f1 = result['f1']

                print(f"[Validation] Epoch {epoch}, F1: {current_f1:.4f}")

                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    no_improve_epochs = 0

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_f1': best_val_f1
                    }, best_model_path)

                    print(f"[Best F1 Updated] Epoch {epoch}, F1: {current_f1:.4f}")
                else:
                    no_improve_epochs += 1
                    print(f"No improvement in F1 for {no_improve_epochs} epoch(s)")

                if no_improve_epochs >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}, best F1: {best_val_f1:.4f}")
                    break