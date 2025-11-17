import torch
import onnxruntime as ort
import numpy as np
import tifffile

ort.set_default_logger_severity(3)

class OnnxModel():
    def __init__(self, model_path, device):
        providers = ['CUDAExecutionProvider']
        self.device = device
        self.session = ort.InferenceSession(model_path, providers=providers, device=f'{device.type}:{device.index}')

    def __call__(self, torch_tensor):
        torch_tensor = torch_tensor.to(self.device)
        x_tensor = torch_tensor.contiguous()

        binding = self.session.io_binding()
        inputname = self.session.get_inputs()[0].name
        outputname = self.session.get_outputs()[0].name

        binding.bind_input(name=inputname, device_type=self.device.type, device_id=self.device.index,
                           element_type=np.float32, shape=tuple(x_tensor.shape),
                           buffer_ptr=x_tensor.data_ptr())

        binding.bind_output(name=outputname)

        self.session.run_with_iobinding(binding)

        y = torch.tensor(binding.copy_outputs_to_cpu()[0])

        return y

def to_tensor(t, device):
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)

    t = t.to(torch.float32)

    # batch, channels, height, width
    while len(t.shape) < 4:
        t = torch.unsqueeze(t, 0)

    t = t.to(device)

    return t

if __name__ == '__main__':

    onnx_file = '/home/yclai/vscode_project/Background_Removal_Net/onnx_models/background_removal_stripe_normalized.onnx'
    device = torch.device('cuda:0')
    model = OnnxModel(onnx_file, device)

    test_image_path = '/home/yclai/vscode_project/Background_Removal_Net/MVTec_AD_dataset/grid_stripe/test/bright_spots/250.tiff'
    image = tifffile.imread(test_image_path)
    image = np.transpose(image, (1, 2, 0))

    cur = image[:, :, 0]
    ref0 = image[:, :, 1]
    ref1 = image[:, :, 2]

    inference_inputs = [cur, ref0, ref1]
    tensor = to_tensor(np.stack(inference_inputs, axis=0), device)
    inf_scores = model(tensor).detach().to('cpu')

    score = np.squeeze(inf_scores[:, 0:1, :, :].numpy())

    print(f"Input shape: {tensor.shape}")
    print(f"Output score shape: {score.shape}")
    print(f"Score range: [{score.min():.6f}, {score.max():.6f}]")
    print(f"Test passed!")

    # Visualization (for internal testing only)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(4, 12))
    im = ax.imshow(score, cmap='hot', vmin=score.min(), vmax=score.max())
    ax.set_title('Anomaly Score')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('onnx_test_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: onnx_test_result.png")



