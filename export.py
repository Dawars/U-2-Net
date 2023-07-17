import torch.jit

from train import WrapperModel

u2net = WrapperModel.load_from_checkpoint("./epoch=77.ckpt", model_name="u2netp")
u2net.cuda()
u2net.eval()

u2net.to_torchscript("u2netp.pth", method="trace")
optimized = torch.jit.optimize_for_inference(u2net.to_torchscript(method="trace"))
optimized.save("u2netp_optimized.pth")
