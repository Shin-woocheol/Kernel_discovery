import os
import subprocess
import sys
from pathlib import Path
from platform import machine

import torch
import numpy as np

from hdbo.benchsuite import settings
from hdbo.benchsuite.benchmark import Benchmark


class Mopta08(Benchmark):

    def __init__(self):
        dim = 124
        super().__init__(
            dim=dim,
            lb=torch.zeros(dim, device=settings.DEVICE, dtype=settings.DTYPE),
            ub=torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE),
        )

        self.sysarch = 64 if sys.maxsize > 2 ** 32 else 32
        self.machine = machine().lower()

        if self.machine == "armv7l":
            assert self.sysarch == 32, "Not supported"
            self._mopta_executable_name = "mopta08_armhf.bin"
        elif self.machine == "x86_64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_executable_name = "mopta08_elf64.bin"
        elif self.machine == "i386":
            assert self.sysarch == 32, "Not supported"
            self._mopta_executable_name = "mopta08_elf32.bin"
        elif self.machine == "amd64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_executable_name = "mopta08_amd64.exe"
        else:
            raise RuntimeError("Machine with this architecture is not supported")

        self._mopta_executable = os.path.abspath(os.path.join(
            Path(__file__).parent.parent, "data", "mopta08", self._mopta_executable_name
        ))
        
        # Ensure the binary is executable
        if os.path.exists(self._mopta_executable) and not os.access(self._mopta_executable, os.X_OK):
            try:
                import stat
                st = os.stat(self._mopta_executable)
                os.chmod(self._mopta_executable, st.st_mode | stat.S_IEXEC)
            except Exception as e:
                print(f"Warning: Could not set execution permission on {self._mopta_executable}: {e}")

        # Use an explicit folder instead of tempfile
        root_data_dir = os.path.join(Path(__file__).parent.parent, "data", "mopta08")
        self.directory_name = os.path.abspath(os.path.join(root_data_dir, f"tmp_mopta_{os.getpid()}"))
        os.makedirs(self.directory_name, exist_ok=True)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Mopta08 benchmark. Handles both single points and batches.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        results = []
        
        for i in range(batch_size):
            xi = x[i]
            # write input to file in dir
            with open(os.path.join(self.directory_name, "input.txt"), "w+") as tmp_file:
                for val in xi:
                    tmp_file.write(f"{val.detach().cpu().numpy()}\n")
            
            # pass directory as working directory to process
            popen = subprocess.Popen(
                self._mopta_executable,
                stdout=subprocess.PIPE,
                cwd=self.directory_name,
            )
            popen.wait()
            
            # read and parse output file
            output_path = os.path.join(self.directory_name, "output.txt")
            if not os.path.exists(output_path):
                results.append(float('inf'))
                continue
                
            with open(output_path, "r") as f:
                lines = f.read().split("\n")
            
            lines = [l.strip() for l in lines if l.strip()]
            output_vals = torch.tensor([float(l) for l in lines], dtype=settings.DTYPE, device=settings.DEVICE)
            
            if len(output_vals) == 0:
                results.append(float('inf'))
                continue
                
            value = output_vals[0]
            constraints = output_vals[1:]
            # see https://arxiv.org/pdf/2103.00349.pdf E.7
            results.append(value + 10 * torch.sum(torch.clip(constraints, min=0, max=None)))

        return torch.tensor(results, dtype=settings.DTYPE, device=settings.DEVICE).unsqueeze(-1)
