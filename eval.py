import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def evaluate_model_with_metrics(model, dataloader, device):
    model.eval()
    y_true_all = []
    y_pred_all = []
    correct_graphs = 0
    total_graphs = 0
    wrong_graph_info = []

    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = batch_data.to(device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
            y = batch_data.y
            pred_y = model.inference(x, edge_index, batch)
            true_y = []
            for i in torch.unique(batch):
                mask = (batch == i)
                true_y.append(y[mask].float())

            graph_ids = getattr(batch_data, "graph_id", [f"graph_{idx}" for idx in range(len(true_y))])
            for i, (pred, true) in enumerate(zip(pred_y, true_y)):
                graph_id = graph_ids[i] if i < len(graph_ids) else f"unknown_{total_graphs}"

                if torch.argmax(pred) == torch.argmax(true):
                    correct_graphs += 1
                else:
                    wrong_graph_info.append((total_graphs, graph_id))

                total_graphs += 1
                y_pred_all.append(pred.cpu())
                y_true_all.append(true.cpu())
    y_pred_all = [torch.argmax(y) for y in y_pred_all]
    y_true_all = [torch.argmax(y) for y in y_true_all]
    y_pred_all = torch.tensor(y_pred_all).numpy()
    y_true_all = torch.tensor(y_true_all).numpy()

    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    precision = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    print(f"[Eval] Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

    return {
        'accuracy': acc,
        'f1': f1,
        'recall': recall,
        'precision': precision
    }