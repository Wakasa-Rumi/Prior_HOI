import json
import torch
import time

def write_log(mesh_loss=torch.tensor([0]), trans_loss=torch.tensor([0]), scale_loss=torch.tensor([0]),
             hand_loss=torch.tensor([0]), contact_loss=torch.tensor([0]), CDC=0, CD=0,
              path=""):
    loss_log = {
        "Mesh loss": mesh_loss.item(),
        "Trans loss": trans_loss.item(),
        "Scale loss": scale_loss.item(),
        "Hand loss": hand_loss.item(),
        "Contact loss": contact_loss.item(),
        "CDC": CDC,
        "CD": CD,
        "time": time.asctime()
    }
    json_str = json.dumps(loss_log, indent=4)
    with open(path, "w", newline="\n") as f:
        f.write(json_str)