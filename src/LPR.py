import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import pickle
import os
from matplotlib.backends.backend_pdf import PdfPages
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, max_per_task=500):
        self.data = []  
        self.max_per_task = max_per_task

    def add_task_data(self, dataset):
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        all_x, all_y = [], []
        for x, y in loader:
            all_x.append(x)
            all_y.append(y)
            if len(torch.cat(all_x)) >= self.max_per_task:
                break
        x_cat = torch.cat(all_x)[:self.max_per_task]
        y_cat = torch.cat(all_y)[:self.max_per_task]
        self.data.append((x_cat, y_cat))  

    def get_loader(self, batch_size=128):
        x_all = torch.cat([x for x, _ in self.data])
        y_all = torch.cat([y for _, y in self.data])
        dataset = list(zip(x_all, y_all))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 0, pin_memory=True)
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=400, num_classes=10, num_layers=3):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x, return_intermediate=False):
        x = x.view(x.size(0), -1)
        z1 = self.net[1](self.net[0](x))
        z2 = self.net[3](self.net[2](z1))
        if return_intermediate:
            return self.net[4](z2), {"fc1": z1, "fc2": z2}
        return self.net(x)
def preprocess_dataset(dataset, degrees):
    rotated_data = []
    for img, label in dataset:
        rotated_img = transforms.functional.rotate(img, angle=degrees)
        rotated_data.append((rotated_img, label))
    return rotated_data
base_transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root='.', train=True, download=True, transform=base_transform)
mnist_test = datasets.MNIST(root='.', train=False, download=True, transform=base_transform)
rotations = [0, 60, 120, 180]
train_datasets = [preprocess_dataset(mnist_train, deg) for deg in rotations]
test_datasets = [preprocess_dataset(mnist_test, deg) for deg in rotations]

def visualize_tasks(datasets, n_samples=49):
    fig, axes = plt.subplots(1, len(datasets), figsize=(len(datasets)*3, 4))
    for i, ds in enumerate(datasets):
        subset = torch.utils.data.Subset(ds, np.random.choice(len(ds), n_samples, replace=False))
        loader = torch.utils.data.DataLoader(subset, batch_size=n_samples, shuffle=False)
        images, _ = next(iter(loader))
        grid = make_grid(images, nrow=int(np.sqrt(n_samples)), pad_value=1)
        axes[i].imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        axes[i].set_title(f"Task {i+1}")
        axes[i].axis('off')
    plt.show()

visualize_tasks(train_datasets)
def evaluate(model, dataset, batch_size=1000):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total
def compute_preconditioners(model, buffer_loader, omega=1.0):
    model.eval()
    with torch.no_grad():
        Z = {"fc1": [], "fc2": [], "fc3": []}
        for x, _ in buffer_loader:
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)
            z1 = model.net[0](x_flat)
            z2 = model.net[2](F.relu(z1))
            Z["fc1"].append(x_flat)
            Z["fc2"].append(z1)
            Z["fc3"].append(z2)  

        preconds = {}
        for k in Z:
            Zmat = torch.cat(Z[k], dim=0)  
            P = torch.eye(Zmat.shape[1], device=device) + omega * (Zmat.T @ Zmat)
            preconds[k] = torch.linalg.inv(P)
        return preconds


def train_with_lpr(model, train_datasets, test_datasets, iters_per_task, lr=0.5, batch_size=256, eval_interval=5,
                   omega=0.1):
    global_iter = 0  
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()
    performance_all_tasks = [[] for _ in test_datasets]

    for task_id, task_data in enumerate(train_datasets):
        print(f"\nTraining on Tasks 1 to {task_id + 1}")

        joint_data = ConcatDataset(train_datasets[:task_id + 1])
        batch_size_to_use = (task_id + 1) * batch_size

        loader = DataLoader(joint_data, batch_size=batch_size_to_use, shuffle=True, drop_last=True, num_workers=0)

        loader_iter = iter(loader)
        subset_loader = DataLoader(task_data, batch_size=300, shuffle=True)
        for x, y in subset_loader:
            replay_buffer.add_task_data(task_data)
            break

        buffer_loader = replay_buffer.get_loader()

        preconds = compute_preconditioners(model, buffer_loader, omega=omega) if task_id > 0 else None

        current_iters = iters_per_task[task_id]
        progress_bar = tqdm(range(current_iters), desc=f"Training Task {task_id + 1}", ncols=100)
        for i in progress_bar:
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output, activations = model(x, return_intermediate=True)
            loss = F.cross_entropy(output, y)
            loss.backward()

            if preconds:
                for name, layer in [("fc1", model.net[0]), ("fc2", model.net[2])]:
                    g = layer.weight.grad.view(layer.weight.shape[0], -1)
                    g_p = g @ preconds[name].T
                    layer.weight.grad.copy_(g_p.view_as(layer.weight.grad))

            optimizer.step()

            if (i + 1) % eval_interval == 0 or i == 0:
                accs = [evaluate(model, testset, batch_size=1000)
                        if idx <= task_id else None
                        for idx, testset in enumerate(test_datasets)]
                for idx, acc in enumerate(accs):
                    if acc is not None:
                        performance_all_tasks[idx].append((global_iter, acc))  
                progress_bar.set_description(" | ".join(
                    [f"Task{i + 1}: {a:.1f}%" for i, a in enumerate(accs) if a is not None]))
            else:
                for idx in range(len(test_datasets)):
                    if idx <= task_id:
                        prev = performance_all_tasks[idx][-1][1] if performance_all_tasks[idx] else 0
                        performance_all_tasks[idx].append((global_iter, prev))
            global_iter += 1  

    return performance_all_tasks


def save_partial_results(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_partial_results(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    omegas_to_try = [0.01, 0.1, 0.5, 1.0, 10.0]
    lrs_to_try = [0.01, 0.1, 0.5, 1.0]
    n_runs = 5
    results_file = "lpr_results.pkl"
    all_performances = load_partial_results(results_file)

    for omega in omegas_to_try:
        for lr in lrs_to_try:
            key = (omega, lr)
            if key not in all_performances:
                all_performances[key] = []

            existing_runs = len(all_performances[key])
            if existing_runs >= n_runs:
                continue

            print(f"\n=== Repeated training for omega = {omega}, lr = {lr} ===")
            for run in range(existing_runs, n_runs):
                print(f"Run {run + 1}/{n_runs}")
                model = MLP().to(device)
                performance = train_with_lpr(
                    model,
                    train_datasets,
                    test_datasets,
                    iters_per_task=[500, 750, 1000, 1250],
                    lr=lr,
                    batch_size=256,
                    eval_interval=5,
                    omega=omega
                )
                all_performances[key].append(performance)
                save_partial_results(results_file, all_performances)  
    def aggregate_runs(perf_runs):
        n_tasks = len(perf_runs[0])
        aggregated = []
        for task_id in range(n_tasks):
            task_curves = [np.array(run[task_id]) for run in perf_runs]
            min_len = min(len(curve) for curve in task_curves)
            task_curves = [curve[:min_len] for curve in task_curves]
            iters = task_curves[0][:, 0]
            accs = np.stack([curve[:, 1] for curve in task_curves])
            mean = np.mean(accs, axis=0)
            std = np.std(accs, axis=0)
            aggregated.append((iters, mean, std))
        return aggregated

    colors = ['blue', 'green', 'red', 'orange']
    task_labels = ['Task 1', 'Task 2', 'Task 3', 'Task 4']
    task_switch_iters = [500, 1250, 2250]

    with PdfPages("lpr_results_plots.pdf") as pdf:
        for omega in omegas_to_try:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.flatten()

            for i, lr in enumerate(lrs_to_try):
                ax = axs[i]
                key = (omega, lr)
                if key not in all_performances or len(all_performances[key]) == 0:
                    ax.set_title(f"LR = {lr} (No data)")
                    ax.axis('off')
                    continue

                perf_runs = all_performances[key]
                aggregated = aggregate_runs(perf_runs)

                for task_id, (iters, mean_acc, std_acc) in enumerate(aggregated):
                    ax.plot(iters, mean_acc, label=task_labels[task_id], color=colors[task_id])
                    ax.fill_between(iters, mean_acc - std_acc, mean_acc + std_acc,
                                    color=colors[task_id], alpha=0.2)

                for j, switch in enumerate(task_switch_iters):
                    ax.axvline(switch, color='gray', linestyle='--', label='Task switch' if j == 0 else None)

                ax.set_title(f"LR = {lr}")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Accuracy (%)")
                ax.set_ylim(0, 100)
                ax.grid(True)
                ax.legend()

            fig.suptitle(f"Mean ± Std Accuracy During LPR Training (ω = {omega}, {n_runs} runs)", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)



