import os 
import numpy as np 
import torch 
import yaml

with open("config.yaml", "r") as f: 
    config = yaml.load(f, Loader=yaml.FullLoader)


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite: 
    
    def __init__(self, B, T, process_rank, num_processes, split, master_process, resume=False, current_step=None, previous_B=None):
        self.B = B 
        # in case you processed with a different step
        self.previous_B = previous_B
        self.T = T 
        self.process_rank = process_rank 
        self.num_processes = num_processes 
        self.master_process = master_process


        assert split in {'train', 'val'}

        # get the shard filenames
        # data_root = "edu_fineweb10B"
        data_root = config['i/o']['dataset_path']
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if self.master_process:
            print(f"found {len(shards)} shards for split {split}")

        if not resume:
            self.reset()
        else: 
            # resume the training from the shard and position you were left
            # we ensure that we expose the model to a diverse set of data rather than the same shard all over again
            self.resume(current_step=current_step)
    
   
    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def resume(self, current_step):
        """Resumes the data loader from a specific training step without storing shard info"""
        
        # Compute how many tokens were processed so far
        tokens_processed = current_step * (self.previous_B * self.T * self.num_processes)

        # Find the correct shard and offset
        tokens_count = 0
        for i, shard in enumerate(self.shards):
            shard_tokens = len(load_tokens(shard))  # Get the number of tokens in this shard
            if tokens_processed < tokens_count + shard_tokens:
                self.current_shard = i
                self.tokens = load_tokens(self.shards[self.current_shard])
                self.current_position = tokens_processed - tokens_count
                break
            tokens_count += shard_tokens  # Accumulate total tokens processed so far

        if self.master_process:
            print(f"Resuming from step {current_step}, shard {self.current_shard}, position {self.current_position}")


    def next_batch(self): 
        B, T = self.B, self.T 

        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)

        self.current_position +=B*T * self.num_processes

        if self.current_position + (B*T * self.num_processes +1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
    
        return x,y 