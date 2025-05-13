import numpy as np

class Cache():
    """ A custom implementation for SMPLer_X pipeline
        Need to run tool/cache/fix_cache.py to fix paths
    """
    def __init__(self, load_path=None):
        if load_path is not None:
            self.load(load_path)

    def load(self, load_path):
        self.load_path = load_path
        self.cache = np.load(load_path, allow_pickle=True)
        self.data_len = self.cache['data_len']
        assert self.data_len == len(self.cache) - 1
        self.cache = None

    @classmethod
    def save(cls, save_path, data_list):
        assert save_path is not None, 'save_path is None'
        data_len = len(data_list)
        cache = {}
        for i, data in enumerate(data_list):
            cache[str(i)] = data
        assert len(cache) == data_len
        # update meta
        cache.update({
            'data_len': data_len,})

        np.savez_compressed(save_path, **cache)
        print(f'Cache saved to {save_path}.')

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.cache is None:
            self.cache = np.load(self.load_path, allow_pickle=True)
        # mapped_idx = self.mapping[idx]
        # cache_data = self.cache[str(mapped_idx)]
        cache_data = self.cache[str(idx)]
        data = cache_data.item()
        return data