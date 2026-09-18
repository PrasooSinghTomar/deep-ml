import torch

def linear_regression_normal_equation(X, y) -> torch.Tensor:
    """
    Solve linear regression via the normal equation using PyTorch.
    X: Tensor or convertible of shape (m,n); y: shape (m,) or (m,1).
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    X = torch.as_tensor(X, dtype = torch.float)
    y = torch.as_tensor(y, dtype = torch.float).reshape(-1,1)

    XtX = X.T @ X
    Xty = X.T @ y

    theta = torch.linalg.solve(XtX, Xty)

    theta = torch.round(theta.flatten() * 10000) / 10000

    return theta 
