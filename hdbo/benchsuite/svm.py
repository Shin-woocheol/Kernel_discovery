import gzip
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from hdbo.benchsuite import settings
from hdbo.benchsuite.benchmark import Benchmark


class SVM(Benchmark):
    """
    The interior benchmark is just the benchmark in the lower-dimensional effective embedding.
    """

    def __init__(
            self,
    ):
        dim = 388
        super().__init__(
            dim=dim,
            lb=torch.zeros(dim, device=settings.DEVICE, dtype=settings.DTYPE),
            ub=torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE),
        )
        self.X_data, self.y_data = self._load_data()
        np.random.seed(388)
        idxs = np.random.choice(np.arange(len(self.X_data)), min(500, len(self.X_data)), replace=False)
        half = len(idxs) // 2
        self._X_train = self.X_data[idxs[:half]]
        self._X_test = self.X_data[idxs[half:]]
        self._y_train = self.y_data[idxs[:half]]
        self._y_test = self.y_data[idxs[half:]]

    def _load_data(self):
        then = time.time()
        hdbo_dir = Path(__file__).parent.parent
        data_folder = os.path.join(hdbo_dir, "data", "svm")
        
        try:
            X = np.load(os.path.join(data_folder, "CT_slice_X.npy"))
            y = np.load(os.path.join(data_folder, "CT_slice_y.npy"))
        except:
            x_gz = os.path.join(data_folder, "CT_slice_X.npy.gz")
            y_gz = os.path.join(data_folder, "CT_slice_y.npy.gz")
            with gzip.GzipFile(x_gz, "r") as fx, gzip.GzipFile(y_gz, "r") as fy:
                X = np.load(fx)
                y = np.load(fy)

        X = MinMaxScaler().fit_transform(X)
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).squeeze()
        now = time.time()
        print(f"Loaded data in {now - then:.2f} seconds")
        return X, y

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both single point and batch
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        results = []
        
        for i in range(batch_size):
            y_np = x[i].cpu().numpy()
            C = 0.01 * (500 ** y_np[387])
            gamma = 0.1 * (30 ** y_np[386])
            epsilon = 0.01 * (100 ** y_np[385])
            length_scales = np.exp(4 * y_np[:385] - 2)

            svr = SVR(gamma=gamma, epsilon=epsilon, C=C, cache_size=1500, tol=0.001)
            svr.fit(self._X_train / length_scales, self._y_train)
            pred = svr.predict(self._X_test / length_scales)
            error = np.sqrt(np.mean(np.square(pred - self._y_test)))
            results.append(error)

        return torch.tensor(results, dtype=settings.DTYPE, device=settings.DEVICE).unsqueeze(-1)
