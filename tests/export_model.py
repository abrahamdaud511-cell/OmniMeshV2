import torch
from omnimesh import OmniMeshV2, ModelConfig

def export_to_onnx(model, output_path="model.onnx"):
    dummy_input = torch.randint(0, 1000, (1, 128))
    torch.onnx.export(model, dummy_input, output_path, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size', 1: 'sequence'}})
    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    config = ModelConfig()
    model = OmniMeshV2(config)
    export_to_onnx(model)
