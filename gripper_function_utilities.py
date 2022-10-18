import torch


class Utilities:
    def __init__(self):
        pass

    def detect_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Working with GPU")
        else:
            device = torch.device('cpu')
            print("Working with CPU")
        return device

    def save_vae_model(self, model):
        torch.save(model.state_dict(), "trained_models/vae_model_gripper.pth")
        print("VAE model has been saved")

    def load_vae_model(self, model):
        model.load_state_dict(torch.load("trained_models/vae_model_gripper.pth"))
        print("VAE model has been loaded")

