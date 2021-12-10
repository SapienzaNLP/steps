import torch

def pad(samples):
    
        batch_size = len(samples)
        max_length = max([len(sample) for sample in samples])
        batch = torch.ones((batch_size, max_length), dtype=torch.int64)
        for i in range(len(samples)):
            for j in range(len(samples[i])):
                batch[i, j] = samples[i][j]

        return batch

def collate_fn(samples):
    keys = samples[0].keys()
    dictionary = {}
    for key in keys:
        lists= []
        for sample in samples:
            lists.append(sample[key])
        padding = pad(lists)
        dictionary[key] = padding
    return dictionary