from functools import partial
import torch
import numpy as np


def gaussian_kernel(x, y, sigma):
    """
    Creates gaussian kernel with sigma bandwidth

    Args:
        x: The first vector
        y: The second vector
        sigma: Gaussian kernel bandwidth

    Returns: Gaussian distance of x and y

    """
    dist_sq = torch.sum((x - y) ** 2, dim=-1)
    kernel = torch.exp(-0.5 * dist_sq / sigma ** 2)
    return kernel

def bandwidth_decorator(function):
    """
    This decorator is only for RKE class is used when the `kernel_bandwidth` is a list.
    """
    def wrap_bandwidth_list(self, *args, **kwargs):
        output = {}
        if self.kernel_bandwidth is not None:  # Gaussian kernel
            for bandwidth in self.kernel_bandwidth:
                self.kernel_function = partial(gaussian_kernel, sigma=bandwidth)
                output[bandwidth] = function(self, *args, **kwargs)
        else:  # Specified kernel
            return function(self, *args, **kwargs)

        return output
    return wrap_bandwidth_list


class RKE:
    def __init__(self, kernel_bandwidth=None, kernel_function=None):
        """
        Define the kernel for computing the Renyi Kernel Entropy score

        Args:
            kernel_bandwidth: The bandwidth to use in gaussian_kernel.
                You should pass either of the `kernel_function` or `kernel_bandwidth` arguments.
            kernel_function: The Kernel function to build kernel matrix,
                default is gaussian_kernel as used in the paper.
        """
        if kernel_function is None and kernel_bandwidth is None:
            raise ValueError('Expected either kernel_function or kernel_bandwidth args')
        if kernel_function is not None and kernel_bandwidth is not None:
            raise ValueError('`kernel_function` is mutually exclusive with `kernel_bandwidth`')

        if kernel_function is None:  # Gaussian kernel
            # Make `kernel_bandwidth` into a list if the input is float or int
            if isinstance(kernel_bandwidth, (float, int)):
                self.kernel_bandwidth = [kernel_bandwidth]
            else:
                self.kernel_bandwidth = kernel_bandwidth
            self.kernel_function = partial(gaussian_kernel, sigma=self.kernel_bandwidth[0])

        else:  # Specified kernel
            self.kernel_bandwidth = None
            self.kernel_function = kernel_function

    
    @bandwidth_decorator
    def compute_rke_mc_frobenius_norm(self, X, block_size=1000):
        """
        Compute the Frobenius norm of the kernel matrix using blocks to reduce memory usage.

        Args:
            X: Input features
            block_size: Size of blocks to process in parallel.

        Returns: Frobenius norm of the kernel matrix
        """
        X = torch.tensor(X, dtype=torch.float32).cuda()
        n_data = X.shape[0]
        sum_frobenius = 0.0
        for i in range(0, n_data, block_size):
            for j in range(0, n_data, block_size):
                X_block = X[i:i+block_size]
                Y_block = X[j:j+block_size]
                kernel_block = self.kernel_function(X_block.unsqueeze(0), Y_block.unsqueeze(1)) ** 2
                sum_frobenius += torch.sum(kernel_block).item()

        return (sum_frobenius / (n_data ** 2))
    
    @bandwidth_decorator
    def compute_rke_mc(self, X, n_samples=1_000_000):
        """
        Computing RKE-MC = exp(-RKE(X))
        Args:
            X: Input features
            n_samples: How many samples to compute k(x_i, x_j).

        Returns: RKE Mode count (RKE-MC)
        """
        X = torch.tensor(X, dtype=torch.float32).cuda()
        n_data = X.shape[0]
        indices = torch.randint(0, n_data, (n_samples, 2), device='cuda')
        similarities = self.kernel_function(X[indices[:, 0]], X[indices[:, 1]]) ** 2
        return 1 / similarities.mean().cpu().item()

    def __compute_relative_kernel(self, X, Y):
        X = torch.tensor(X, dtype=torch.float32).cuda()
        Y = torch.tensor(Y, dtype=torch.float32).cuda()
        output = self.kernel_function(X.unsqueeze(0), Y.unsqueeze(1))
        return output / np.sqrt(X.shape[0] * Y.shape[0])

    @bandwidth_decorator
    def compute_rrke(self, X, Y, x_samples=500, y_samples=None):
        if y_samples is None:
            y_samples = x_samples

        X = torch.tensor(X, dtype=torch.float32).cuda()
        Y = torch.tensor(Y, dtype=torch.float32).cuda()

        k_xy = self.__compute_relative_kernel(X[:x_samples], Y[:y_samples])
        svds = torch.svd(k_xy, compute_uv=False).S
        sum_svds = torch.sum(svds)
        return -torch.log(sum_svds**2).cpu().item()
    
