import os 
import torch 
from torch.nn import functional as F 
from torch.distributed import init_process_group, destroy_process_group 
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken 
from refactored_data_loader import DataLoaderLite
from refactored_gpt2 import GPT, GPTConfig
import math 
from hellaswag import render_example, iterate_examples
import time  
from datetime import datetime
import yaml
import os
import subprocess

with open("config.yaml", "r") as f: 
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)


# wandb logging
wandb_log = config_yaml['wandb']['is_logging'] # disabled by default
wandb_project = config_yaml['wandb']['project_name']
wandb_id_run = config_yaml['wandb']['id']
wandb_run_name = 'gpt2' + '_run_' + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
resumed_training = config_yaml['i/o']['resumed_training']
checkpoint_path = config_yaml['i/o']['checkpoint_path'] # the path to the previous model 

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm



# DDP = distributed data parallel
ddp = int(os.environ.get('RANK', -1)) != -1 

if ddp: 
    assert torch.cuda.is_available(), "for now I think we need CUDA for DDP"

    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE']) 
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 #from this one we load checkpoints and we do logging
else: 
    # vanilla run
    ddp_rank = 0
    ddp_local_rank = 0 
    ddp_world_size = 1 
    master_process = True 

    device ="cpu"

    if torch.cuda.is_available(): 
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
        device = "mps" 
    
    print(f"Using device: {device}")


device_type = "cuda" if device.startswith("cuda") else "cpu"


torch.manual_seed(1337)

if torch.cuda.is_available(): 
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

# torchrun --standalone --nproc_per_node=8 train_gpt2.py


torch.set_float32_matmul_precision('high')

total_batch_size = 524288
B = 8 # we need to use 8 
previous_B = 32 # in case you need to load a checkpoint and you used a different batch size
T = 1024 
assert total_batch_size % (B * T *ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # we take into account the number of processes

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")



# Resuming the checkpoint for the shards

start_step = None

if resumed_training: 
    # Load the checkpoint for obtaining the step 
    # Not the cleanest way to do it, since you reload the checkpoint one more time for the model and the optimizer
    # I'll refactor this later on 
    checkpoint = torch.load(checkpoint_path, map_location=device) 
    start_step = checkpoint['step']


# We resume only the shards for the train set, since we have a single shard for validation 
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", master_process=master_process, resume=resumed_training, current_step=start_step, previous_B=previous_B)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", master_process=master_process)


# create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device) 
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. 

if use_compile:
    model = torch.compile(model)

raw_model = model.module if ddp else model # always containts the raw unwrapped model


if ddp: 
    model = DDP(model, device_ids=[ddp_local_rank])


max_lr = 6e-4 
min_lr = max_lr*0.1 
warmup_steps = 715 
max_steps = 19073 
period = 200 # after how many steps we perform validation 

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    if wandb_id_run == "":
        wandb.init(project=wandb_project, 
                mode ='online',
                name=wandb_run_name, 
                config={
                    "learning_rate": 6e-4, 
                    "weight_decay": 0.1, 
                    "warmup_steps": warmup_steps, 
                    "max_steps": max_steps, 
                    "batch_size": B, 
                    "context_size": T, 
                    "device": device,                    
                }
        )
    else: 
        wandb.init(project=wandb_project, 
                mode ='online',
                resume='must',
                id=wandb_id_run, 
        )


#optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)


start_step = 0

# we continue the training from where we were left
if resumed_training: 
    print("We are resuming the training!!")
    # checkpoint = torch.load(checkpoint_path, map_location=device) 
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer']) 

    #we resume a training, so we skip a couple of steps 
    #the current step is affecting the lr scheduler
    start_step = checkpoint['step']


# create the log directory we will write checkpoints to and log to
log_dir = config_yaml['i/o']['log_path']
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass


previous_train_loss = 0.0

for step in range(start_step, max_steps): 
    t0 = time.time() 
    last_step = (step == max_steps - 1)


    # normal validation 
    # once in a while evaluate our validation loss
    if step % period == 0 or last_step:
        print("Validation time!")
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            # we should ignore if step = 0 ; fix it afterwards
            if step > 0 and (step % 200 == 0 or last_step):
                print('Saving data and the model')
                if wandb_log:
                    wandb.log({
                        "iter": step,
                        "train/loss": previous_train_loss,
                        "val/loss": val_loss_accum.item(),
                        "lr": optimizer.param_groups[0]['lr']
                    })

                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                    # try to save the data from the data loader needed to continue the training
                }

                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

                # removes the second oldest checkpoint
                # it ensures that you do not overload your disk with too many checkpoints (useful if you work in cloud)
                old_checkpoint = os.path.join(log_dir, f"model_{(step-2*200):05d}.pt")
                subprocess.run(["rm", old_checkpoint], check=False)
                

    
    #normal hellaswag eval 
    # once in a while evaluate hellaswag
    if (step % period == 0 or last_step) and (not use_compile):
        print("HellaSwag time!")
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
    

    # check some generations
    # once in a while generate from the model (except step 0, which is noise)
    if ((step >= 0 and step % period == 0) or last_step) and (not use_compile):
        print("Generation time!")
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    # Training Procedure

    optimizer.zero_grad() 
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x,y = train_loader.next_batch() 
        x,y = x.to(device), y.to(device)

        # Don't forget to apply autocast when you move to cloud GPU
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x,y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()


        # it should synchronise the processes on the final microstep (we need to gather the error)
        if ddp: 
            model.require_backward_grad_sync = (micro_step == grad_accum_steps -1)

        loss.backward()

    if ddp: 
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups: 
        param_group['lr'] = lr 

    optimizer.step() 

    if device_type == "cuda":
        torch.cuda.synchronize() 
    
    if device_type == "mps":
        torch.mps.synchronize() 

    t1 = time.time() 
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed/ dt

    previous_train_loss = loss_accum.item()


    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp: 
    destroy_process_group()