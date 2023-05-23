import torch
from torch.utils.data import Dataset, DataLoader
from .utils import seed_worker, get_num_workers

class GLUEDataset(Dataset):
    """ GLUE dataset for single task fine-tuning PLMs """    
    def __init__(self, df):
        super().__init__()
        self.input_ids = df['input_ids']
        self.attention_mask = df['attention_mask']
        self.labels = df['label']
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.input_ids[index]),
            torch.tensor(self.attention_mask[index]),
            torch.tensor(self.labels[index])
        )
    
    def __len__(self):
        return len(self.input_ids)
    
def load_glue_data_loader(ds, batch_size, shuffle=False):
    """ GLUE data loader for single task fine-tuning PLMs """
    num_workers = get_num_workers()
    train_sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=shuffle)\
        if torch.cuda.device_count() >= 1 else None
    data_loader = DataLoader(ds,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             sampler=train_sampler,
                             worker_init_fn=seed_worker)
    return data_loader