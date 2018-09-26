import numpy as np
from torch.cuda import FloatTensor as CUDATensor
from visdom import Visdom


_WINDOW_CASH = {}


def _vis(env='main'):
    return Visdom(env=env)


def visualize_images(tensor, name, env='main', w=400, h=400):
    '''Plot images contained in [tensor] to visdom-server.'''

    # If needed, set tensor back to cpu
    tensor = tensor.cpu() if isinstance(tensor, CUDATensor) else tensor

    # Plot the images
    _WINDOW_CASH[name] = _vis(env).images(
        tensor.numpy(), win=_WINDOW_CASH.get(name), nrow=8, opts=dict(title=name, width=w, height=h)
    )


def visualize_scalars(scalars, names, title, iteration, env='main', ylabel=None):
    '''Continually update line-plot with numbers arriving in [scalars].'''
    assert len(scalars) == len(names)

    # Convert scalar tensors to numpy arrays.
    scalars, names = list(scalars), list(names)
    scalars = [s.cpu() if isinstance(s, CUDATensor) else s for s in scalars]
    scalars = [s.numpy() if hasattr(s, 'numpy') else np.array([s]) for s in scalars]
    num = len(scalars)
    X = np.column_stack(np.array([iteration] * num)) if (num>1) else np.array([iteration] * num)
    Y = np.column_stack(scalars) if (num>1) else scalars[0]

    # Plotting options
    options = dict(
        fillarea=False, legend=names, width=400, height=400,
        xlabel='Iterations', ylabel=title if (ylabel is None) else ylabel, title=title,
        marginleft=30, marginright=30, marginbottom=80, margintop=30,
    )

    # Update plot (or start new one if not yet present)
    if title in _WINDOW_CASH:
        _vis(env).updateTrace(X=X, Y=Y, win=_WINDOW_CASH[title], opts=options)
    else:
        _WINDOW_CASH[title] = _vis(env).line(X=X, Y=Y, opts=options)
