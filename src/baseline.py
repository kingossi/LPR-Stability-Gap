import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from PIL import Image
import os
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=400, num_classes=10, num_layers=3):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=0, pin_memory=True)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

iters_per_task = [500, 750, 1000, 1250]


def train_and_evaluate_multi(model, trainset, testsets, iters, lr=0.1, batch_size=256,
                             test_size=1000, eval_interval=1):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    performance_by_task = [[] for _ in testsets]

    loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=True, drop_last=True,
                                         num_workers=0, pin_memory=True)
    loader_iter = iter(loader)

    progress_bar = tqdm(range(iters), desc="Training", ncols=100)

    for i in progress_bar:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

        if (i + 1) % eval_interval == 0 or i == 0:
            accs = [evaluate(model, testset, batch_size=test_size) for testset in testsets]
            for acc_list, acc in zip(performance_by_task, accs):
                acc_list.append(acc)
            progress_bar.set_description(f"Loss: {loss.item():.3f} | Accs: {accs}")
        else:
            for acc_list in performance_by_task:
                acc_list.append(acc_list[-1] if acc_list else 0)

    return performance_by_task


results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

lrs_to_try = [0.01, 0.1, 0.5, 1.0]
repeats = 5
batch_size = 256

combined_results = {}

for lr in lrs_to_try:
    combined_results[lr] = []
    for run in range(repeats):
        print(f"\nRunning LR={lr}, Repeat={run+1}")
        model = MLP().to(device)
        performance_all_tasks = [[] for _ in rotations]

        for task_id, num_iters in enumerate(iters_per_task):
            joint_data = torch.utils.data.ConcatDataset(train_datasets[:task_id+1])
            testsets = test_datasets[:task_id+1]

            print(f"\nTraining on Tasks 1 to {task_id+1} with LR={lr}, Iters={num_iters}")
            batch_size_to_use = (task_id + 1) * batch_size

            task_performances = train_and_evaluate_multi(
                model, joint_data, testsets,
                iters=num_iters, lr=lr, batch_size=batch_size_to_use
            )

            for i in range(len(rotations)):
                if i <= task_id:
                    performance_all_tasks[i].extend(task_performances[i])
                else:
                    performance_all_tasks[i].extend([None] * num_iters)

        combined_results[lr].append(performance_all_tasks)


final_output_path = os.path.join(results_dir, 'baseline_results.pkl')
with open(final_output_path, 'wb') as f:
    pickle.dump(combined_results, f)

print(f"\nSaved all baseline results to: {final_output_path}")
