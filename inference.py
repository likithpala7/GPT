import torch
from v2 import decode, BigramLanguageModel

model = BigramLanguageModel()
model.load_state_dict(torch.load("gpt_model.pth"))
model.to("cuda")


context = torch.zeros((1, 1), dtype=torch.long, device="cuda")
with open("out.txt", "a") as f:
    f.write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))