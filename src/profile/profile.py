import torch
from torch.profiler import profile, ProfilerActivity, schedule
from tqdm import trange


def profile_model(model, input_size=(1, 3, 512, 512), input_dtype=torch.bfloat16, device='cuda'):
    example_tensor = torch.ones(input_size, dtype=input_dtype, device=device)
    model.to(device)
    model.eval()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            skip_first=1,
            wait=1,
            warmup=3,
            active=2,
            repeat=1,
        ),
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            for _ in trange(7, desc="Profiling"):
                model(example_tensor)
                prof.step()
    
    return prof
