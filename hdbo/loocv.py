import torch
import gpytorch
import math
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Define the GPyTorch Model with an RBF Kernel
# ------------------------------------------------------------------
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Using RBF Kernel (Squared Exponential) as requested
        # Wrapped in a ScaleKernel to parameterize the signal variance
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ------------------------------------------------------------------
# 2. LOOCV Helper Function
# ------------------------------------------------------------------
def get_loocv_parameters(model, likelihood, train_x, train_y):
    """
    Computes the Leave-One-Out Cross-Validation (LOOCV) predictive 
    mean and variance for all training points in O(n^3) time using Cholesky.
    """
    output_dist = likelihood(model(train_x))
    prior_mean = output_dist.mean
    K_noisy = output_dist.covariance_matrix 
    
    y_centered = (train_y - prior_mean).unsqueeze(-1)
    n = train_x.size(0)

    # Numerically stable inversion using Cholesky Decomposition
    L = torch.linalg.cholesky(K_noisy)
    
    eye = torch.eye(n, dtype=train_x.dtype, device=train_x.device)
    K_inv = torch.cholesky_solve(eye, L, upper=False)
    alpha = torch.cholesky_solve(y_centered, L, upper=False).squeeze(-1)

    K_inv_diag = torch.diag(K_inv)

    # Compute LOOCV Predictive Variance and Mean
    var_loo = 1.0 / K_inv_diag
    mean_loo = train_y - (alpha / K_inv_diag) 
    
    return mean_loo, var_loo


# ------------------------------------------------------------------
# 3. Custom Loss Functions (NLPD and CRPS)
# ------------------------------------------------------------------
def loocv_nlpd_loss(model, likelihood, train_x, train_y):
    """
    Computes the LOOCV Negative Log Predictive Density (NLPD).
    Heavily penalizes confidently wrong predictions (sensitive to outliers).
    """
    mean_loo, var_loo = get_loocv_parameters(model, likelihood, train_x, train_y)
    
    # Formula: 0.5 * log(2 * pi * var_loo) + (y - mean_loo)^2 / (2 * var_loo)
    nlpd_i = 0.5 * torch.log(2 * math.pi * var_loo) + (0.5 * (train_y - mean_loo)**2 / var_loo)
    
    return nlpd_i.mean()


def rank_weighted_loss(model, likelihood, train_x, train_y):
    """
    Computes the pointwise Negative Log Likelihood (NLL) on the training set 
    with rank-based reweighting, giving higher weight to high y values.
    """
    with torch.no_grad():
        output = likelihood(model(train_x))
        mean = output.mean
        var = output.variance
        
    # Base pointwise NLL
    nll_i = 0.5 * torch.log(2 * math.pi * var) + (0.5 * (train_y - mean)**2 / var)
    
    # Rank-based reweighting from paper: w \propto 1 / (k * N + rank(y))
    # where rank(y) is the descending rank (0 for the highest/best y)
    N = train_y.shape[0]
    k = 0.01
    
    # argsort on negative train_y gives rank 0 to the highest value
    ranks = (-train_y).argsort().argsort().float()
    
    weights = 1.0 / (k * N + ranks)
    
    return (nll_i * weights).mean()


def loocv_crps_loss(model, likelihood, train_x, train_y):
    """
    Computes the LOOCV Continuous Ranked Probability Score (CRPS).
    Measures the distance between predicted CDF and observed step function.
    More robust to extreme outliers than NLPD.
    """
    mean_loo, var_loo = get_loocv_parameters(model, likelihood, train_x, train_y)
    
    sigma_loo = torch.sqrt(var_loo)
    z = (train_y - mean_loo) / sigma_loo

    # Standard normal PDF (phi) and CDF (Phi)
    inv_sqrt_2pi = 1.0 / math.sqrt(2 * math.pi)
    phi = inv_sqrt_2pi * torch.exp(-0.5 * z**2)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

    # Analytical CRPS for a Gaussian distribution (to be minimized)
    crps_i = sigma_loo * (z * (2 * Phi - 1.0) + 2 * phi - 1.0 / math.sqrt(math.pi))
    
    return crps_i.mean()


# ------------------------------------------------------------------
# 4. Execution and Optimization Loop
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Generate some dummy data (e.g., clustered data where MLL might fail)
    torch.manual_seed(42)
    train_x = torch.cat([torch.linspace(0, 0.3, 10), torch.linspace(0.8, 1.0, 5)])
    
    # Adding a massive outlier to demonstrate CRPS robustness
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.1
    train_y[7] += 3.0  # Outlier injected here!

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Toggle this flag to compare optimizations
    USE_CRPS = True
    loss_name = "CRPS" if USE_CRPS else "NLPD"

    print(f"Starting Optimization using LOOCV {loss_name} (with an injected outlier)...")
    training_iterations = 100

    for i in range(training_iterations):
        optimizer.zero_grad()
        
        if USE_CRPS:
            loss = loocv_crps_loss(model, likelihood, train_x, train_y)
        else:
            loss = loocv_nlpd_loss(model, likelihood, train_x, train_y)
        
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            lengthscale = model.covar_module.base_kernel.lengthscale.item()
            noise = likelihood.noise.item()
            print(f"Iter {i+1:3d}/{training_iterations} - Loss ({loss_name}): {loss.item():.4f} "
                  f"- Lengthscale: {lengthscale:.4f} - Noise: {noise:.4f}")

    print("\nOptimization Complete.")