import torch

class TensorDatasetGpu:
    def __init__(self, x_dict, y, device):
        self.x_cpu = x_dict
        self.y_cpu = y
        self.device = torch.device(device)

        self.x_gpu = self.move_to_device(self.x_cpu, self.device)
        self.y_gpu = self.y_cpu.to(self.device)

        self.n = y.size(0)

    @staticmethod
    def move_to_device(data, device):
        if isinstance(data, dict):
            return {k: TensorDatasetGpu.move_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            raise TypeError(f"Unsupported type: {type(data)}")

    def _shuffle_dict(self, data, idx):
        if isinstance(data, dict):
            return {k: self._shuffle_dict(v, idx) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data[idx]
        else:
            raise TypeError(f"Unsupported type: {type(data)}")

    def _slice_dict(self, data, start, end):
        if isinstance(data, dict):
            return {k: self._slice_dict(v, start, end) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data[start:end]
        else:
            raise TypeError(f"Unsupported type: {type(data)}")

    def shuffle(self):
        idx = torch.randperm(self.n)
        self.x_cpu = self._shuffle_dict(self.x_cpu, idx)
        self.y_cpu = self.y_cpu[idx]
        self.x_gpu = self.move_to_device(self.x_cpu, self.device)
        self.y_gpu = self.y_cpu.to(self.device)

    def get_batch(self, batch_size, i):
        start, end = i * batch_size, min((i + 1) * batch_size, self.n)
        return self._slice_dict( self.x_gpu, start, end), self.y_gpu[start:end]

    def iter_batches(self, batch_size, shuffle_each_epoch=True):
        if shuffle_each_epoch:
            self.shuffle()
        num_batches = (self.n + batch_size - 1) // batch_size
        for i in range(num_batches):
            yield self.get_batch(batch_size, i)

    def get_data_gpu(self):
        return self.x_gpu, self.y_gpu