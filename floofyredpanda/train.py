import os
import torch
from torch import nn as nn_module
from torch.utils.data import TensorDataset, DataLoader
from .nn import _raw_to_tensor

def train(
        raw_inputs,
        raw_outputs,
        layers=None,                       # list of hidden sizes; required only if model is None
        activation='relu',                 # 'relu' or 'tanh'
        dropout=0.0,                       # dropout probability (0.0 = no dropout)
        optimiser_cls=torch.optim.Adam,    # optimiser class
        optimiser_kwargs=None,             # dict of extra optimiser args (e.g. {'weight_decay':1e-5})
        lr=1e-3,                           # learning rate (common to default optimiser_kwargs)
        criterion_cls=nn_module.MSELoss,   # loss function class
        criterion_kwargs=None,             # extra args for loss fn
        epochs=10,
        batch_size=32,
        shuffle=True,
        device=None,
        model=None                         # existing nn.Module to continue training
):
    """
    Train (or fine-tune) a simple feed-forward net.

    All extra behaviours are configurable, but you only ever need to pass the few you care about.

    Args:
      raw_inputs, raw_outputs : any raw data that _raw_to_tensor() can handle
      layers                   : [int,…] hidden sizes; only if model is None
      activation               : 'relu' or 'tanh'
      dropout                  : float in [0,1), dropout after each hidden layer
      optimiser_cls            : torch.optim.XXX class (default Adam)
      optimiser_kwargs         : dict of optimiser args (e.g. {'weight_decay':1e-5})
      lr                       : learning rate (merged into optimiser_kwargs)
      criterion_cls            : loss class (default MSELoss)
      criterion_kwargs         : dict of loss-fn args
      epochs, batch_size, shuffle: usual DataLoader/loop params
      device                   : 'cpu' or 'cuda'; auto-chosen if None
      model                    : existing model to resume training (optional)

    Returns:
      the trained torch.nn.Module
    """

    # sanity
    if len(raw_inputs) != len(raw_outputs):
        raise ValueError("raw_inputs and raw_outputs must be same length")

    # device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device) if model is not None else None

    # raw → tensor
    X = torch.stack([_raw_to_tensor(x).flatten() for x in raw_inputs]).to(device)
    Y = torch.stack([_raw_to_tensor(y).flatten() for y in raw_outputs]).to(device)

    in_dim, out_dim = X.size(1), Y.size(1)

    # build new if needed
    if model is None:
        if layers is None:
            raise ValueError("Must specify layers when model is None.")
        dims = [in_dim] + layers + [out_dim]

        seq = []
        Act = nn_module.ReLU if activation=='relu' else nn_module.Tanh
        for i in range(len(dims)-1):
            seq.append(nn_module.Linear(dims[i], dims[i+1]))
            # add activation+dropout after every hidden layer
            if i < len(dims)-2:
                seq.append(Act())
                if dropout>0.0:
                    seq.append(nn_module.Dropout(p=dropout))

        model = nn_module.Sequential(*seq).to(device)

    # optimiser & loss
    optim_kwargs = dict(optimiser_kwargs or {})
    optim_kwargs.setdefault('lr', lr)
    optimiser = optimiser_cls(model.parameters(), **optim_kwargs)

    crit_kwargs = dict(criterion_kwargs or {})
    criterion = criterion_cls(**crit_kwargs)

    # data loader
    loader = DataLoader(TensorDataset(X, Y),
                        batch_size=batch_size,
                        shuffle=shuffle)

    # training loop
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for xb, yb in loader:
            optimiser.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(loader.dataset)
        print(f"epoch {epoch}/{epochs} – loss: {avg:.4f}")

    return model
