import torch
device = torch.device('cuda:0')
torch.cuda.empty_cache()
# создаём "модельную" базовую память
base = torch.empty(1024, 1024, device=device)  # пример большой тензор

# --- тест 1: ephemeral буферы ---
torch.cuda.reset_peak_memory_stats()
tmp_list = [torch.empty_like(base) for _ in range(4)]   # временные буферы
cat = torch.cat(tmp_list, dim=1)                       # ещё одна временная аллокация
peak_ephemeral = torch.cuda.max_memory_allocated()
del tmp_list, cat
torch.cuda.synchronize()

# --- тест 2: preallocated (reused) ---
torch.cuda.reset_peak_memory_stats()
pre = [torch.empty_like(base) for _ in range(4)]       # аллоцируем заранее (одноразово)
out = torch.empty(base.size(0), base.size(1)*4, device=device)
for i in range(4):
    out[..., i*base.size(1):(i+1)*base.size(1)].copy_(pre[i])
peak_preallocated = torch.cuda.max_memory_allocated()

print("peak ephemeral:", peak_ephemeral)
print("peak preallocated:", peak_preallocated)
