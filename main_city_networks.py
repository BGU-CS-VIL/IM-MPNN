import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from city_networks.citynetworks import CityNetwork
from models.gcn import GCN
from models.immpnn import IMMPNN


def parse_args():
    parser = argparse.ArgumentParser(description='Train GNN model on City Networks dataset')
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='paris',
                      choices=['paris', 'shanghai', 'la', 'london'],
                      help='dataset name (default: paris)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20000,
                      help='number of epochs to train (default: 20000)')
    parser.add_argument('--batch-size', type=int, default=20000,
                      help='batch size for NeighborLoader (default: 20000)')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                      help='number of warmup epochs (default: 2000)')
    parser.add_argument('--lr', type=float, default=1e-3,
                      help='learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                      help='weight decay (default: 1e-5)')
    
    # Sampling parameters
    parser.add_argument('--num-neighbors', type=int, default=16,
                      help='number of neighbors to sample at each step (default: 16)')
    parser.add_argument('--num-hops', type=int, default=16,
                      help='number of hops to consider in neighborhood sampling (default: 16)')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='imgcn',
                      choices=['gcn', 'imgcn'],
                      help='model type (default: imgcn)')
    parser.add_argument('--hidden-channels', type=int, default=32,
                      help='number of hidden channels (default: 32)')
    parser.add_argument('--model-layers', type=int, default=16,
                      help='number of model layers (default: 16)')
    parser.add_argument('--dropout', type=float, default=0.2,
                      help='dropout rate (default: 0.2)')
    parser.add_argument('--residual', action='store_true',
                      help='use residual connections')
    
    # IMGCN specific parameters
    parser.add_argument('--scales', type=int, default=4,
                      help='number of scales for IMGCN (default: 4)')
    
    # GCN specific parameters
    parser.add_argument('--use-mlp-enc-dec', action='store_true',
                      help='use MLP encoder/decoder for GCN')
    
    # Seed for reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    
    # Wandb parameters
    parser.add_argument('--wandb', action='store_true',
                        help='enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='city-networks',
                        help='wandb project name (default: city-networks)')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='wandb entity name (default: None)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='name for the wandb run (default: None)')
    
    return parser.parse_args()


def compute_dataset_stats(data, device):
    """Compute mean and std of the training data."""
    train_data = data.x[data.train_mask]
    mean = train_data.mean(dim=0, keepdim=True).to(device)
    std = train_data.std(dim=0, keepdim=True).to(device)
    return mean, std


def create_model(args, in_channels, out_channels):
    """Create model based on arguments."""
    if args.model == 'gcn':
        return GCN(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=out_channels,
            num_layers=args.model_layers,
            dropout=args.dropout,
            residual=args.residual,
            use_mlp_enc_dec=args.use_mlp_enc_dec
        )
    else:  # imgcn
        return IMMPNN(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=out_channels,
            num_layers=args.model_layers,
            scales=args.scales,
            dropout=args.dropout,
            residual=args.residual
        )


def create_loaders(data, batch_size, num_neighbors, num_hops):
    """Create train, validation and test loaders with specified neighborhood parameters."""
    neighbor_config = [num_neighbors] * num_hops
    
    train_loader = NeighborLoader(
        data,
        num_neighbors=neighbor_config,
        batch_size=batch_size,
        shuffle=True,
        input_nodes=data.train_mask,
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=neighbor_config,
        batch_size=batch_size,
        shuffle=False,
        input_nodes=data.val_mask,
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=neighbor_config,
        batch_size=batch_size,
        shuffle=False,
        input_nodes=data.test_mask,
    )
    
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, data_mean, data_std, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch.x = (batch.x - data_mean) / data_std
        
        out = model(batch)
        out = out[:batch.batch_size] 
        y = batch.y[:batch.batch_size]

        optimizer.zero_grad()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, data_mean, data_std, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(device)
        batch.x = (batch.x - data_mean) / data_std
        out = model(batch)
        
        preds = out[:batch.batch_size].argmax(dim=-1).cpu()
        labels = batch.y[:batch.batch_size].cpu()

        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return accuracy_score(all_labels, all_preds)


def main():
    args = parse_args()

    if args.wandb and not WANDB_AVAILABLE:
        raise ImportError("wandb is not installed but --wandb flag was provided. ")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.wandb:
        run_name = args.run_name if args.run_name else f"{args.model}_{args.dataset}_{args.seed}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)
        )

    print(f"Loading {args.dataset} dataset...")
    dataset = CityNetwork(root="./city_networks", name=args.dataset)
    data = dataset[0]

    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes

    # Create data loaders with specified parameters
    train_loader, val_loader, test_loader = create_loaders(
        data, 
        args.batch_size, 
        args.num_neighbors, 
        args.num_hops
    )

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with specified parameters
    model = create_model(args, in_channels, out_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Compute dataset statistics
    data_mean, data_std = compute_dataset_stats(data, device)

    # Training loop
    best_val_acc = 0.0
    best_test_acc = 0.0

    print(f"\nStarting training with parameters:")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Model parameters:")
    print(f"  - Hidden channels: {args.hidden_channels}")
    print(f"  - Number of layers: {args.model_layers}")
    print(f"  - Dropout: {args.dropout}")
    print(f"  - Residual connections: {args.residual}")
    if args.model == 'imgcn':
        print(f"  - Scales: {args.scales}")
    else:
        print(f"  - Use MLP: {args.use_mlp_enc_dec}")
    print(f"Training parameters:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Weight decay: {args.weight_decay}")
    print(f"Neighborhood sampling:")
    print(f"  - {args.num_neighbors} neighbors for {args.num_hops} hops")
    
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, 
                             data_mean, data_std, device)

        if args.wandb:
            wandb.log({"train/loss": loss, "train/lr": optimizer.param_groups[0]['lr']}, step=epoch)

        if epoch % 100 == 0:
            val_acc = evaluate(model, val_loader, data_mean, data_std, device)
            test_acc = evaluate(model, test_loader, data_mean, data_std, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                checkpoint_path = f'best_model_{args.dataset}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'test_acc': best_test_acc,
                }, checkpoint_path)

            print(f'Epoch {epoch:03d} | '
                  f'Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} '
                  f'(Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f})')

            if args.wandb:
                wandb.log({
                    "val/accuracy": val_acc,
                    "test/accuracy": test_acc,
                    "best/val_accuracy": best_val_acc,
                    "best/test_accuracy": best_test_acc
                }, step=epoch)

    print("Training completed.")
    print(f'Best Val Acc: {best_val_acc:.4f}, corresponding Test Acc: {best_test_acc:.4f}')

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
