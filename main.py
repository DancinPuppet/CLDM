import torch
import numpy as np
import os
from glob import glob
import random
import create_graphs
from args import Args
from data import GraphGroupDataset
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch_geometric.data import Data, Batch
from model import CLDMModel
from train import train_CLDM
from eval import evaluate_model_with_metrics
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
SEED = 123
GRAPH_SEED = 123


def worker_init_fn(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def custom_collate_fn(batch):
    Batch_data = Batch.from_data_list(batch)
    return Batch_data


def custom_collate_fn1(batch):

    data_list = [item[0] for item in batch] 
    node_mapping_list = [item[1] for item in batch] 
    Batch_data = Batch.from_data_list(data_list)
    return Batch_data, node_mapping_list


def dataloader_build(dataset, args_):

    dataloader_ = DataLoader(dataset, batch_size=args_.batch_size, shuffle=True,
                             num_workers=args_.num_workers, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)
    return dataloader_

def test_best_model(args, model, device, dataset_test):

    print("=== Start Testing Best Model ===")

    best_model_path = os.path.join(
        args.save_path,
        f"{args.model_name}_{args.graph_type}_best_f1.pt"
    )

    test_loader = dataloader_build(dataset_test, args)

    print(f"\n[Eval] Loading model from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        metrics = evaluate_model_with_metrics(model, test_loader, device)

        print("\n=== Test Result ===")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"F1       : {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")

def sample_one_batch(dataset, batch_size):
    indices = random.sample(range(len(dataset)), batch_size)
    subset = Subset(dataset, indices)
    dataloader_temp_ = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    return dataloader_temp_


if __name__ == "__main__":
    set_seed(SEED)
    args = Args()
    print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id < torch.cuda.device_count() else 'cpu')
    graphs = create_graphs.data(args, load_exited_graphs=True)
    local_graph_random = random.Random(GRAPH_SEED)
    local_graph_random.shuffle(graphs)
    print("First 5 graph IDs after shuffle:", [g.id for g in graphs[:5]])
    graphs_len = len(graphs)
    graphs_train = graphs[0:int(0.8 * graphs_len)]
    graphs_validate = graphs[int(0.8 * graphs_len):int(0.9 * graphs_len)]
    graphs_test = graphs[int(0.9 * graphs_len):]
    print(f'total graph num:{graphs_len}, training graph num:{len(graphs_train)}, '
          f'validate graph num:{len(graphs_validate)}, test graph num:{len(graphs_test)}')
    print(f'max number node: {args.max_num_node}')
    dataset_train = GraphGroupDataset(graphs_train, args)
    dataloader = dataloader_build(dataset_train, args)
    dataset_validate = GraphGroupDataset(graphs_validate, args)
    dataset_test = GraphGroupDataset(graphs_test, args)
    dataloader_temp = dataloader_build(dataset_validate, args)
    if args.model_name == "CLDM":
        model = CLDMModel(args).to(device)
    print(f'model name: {args.model_name}')
    if args.load_model:
        test_best_model(args, model, device, dataset_test)
    else:
        print('Start training...')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=args.lr * 0.1)
        if args.model_name == "CLDM":
            train_CLDM(args, dataloader, dataloader_temp, model, optimizer, scheduler, device)
        test_best_model(args, model, device, dataset_test)
