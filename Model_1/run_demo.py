import torch
from torch.utils.data import DataLoader
from poisson_tnr.data import PoissonDataset, train_val_split
from poisson_tnr.model import ModelA, ModelB
from poisson_tnr.train import stagewise_train
from poisson_tnr.eval import compute_psnr


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_dir = 'TNRD-Codes/TrainingCodes4denoising/FoETrainingSets180'
    train_paths, val_paths = train_val_split(image_dir)
    dataset = PoissonDataset(train_paths, patch_size=64, peak=60.0,
                             patches_per_image=1, augment=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    modelA = ModelA(num_stages=3, num_filters=8, kernel_size=5).to(device)
    criterion = torch.nn.MSELoss()

    for stage_idx in range(3):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, modelA.parameters()), lr=1e-3)
        print(f'Training ModelA stage {stage_idx} for 1 epoch...')
        loss = stagewise_train(modelA, loader, optimizer, criterion, device, stage_idx=stage_idx)
        print(f'stage {stage_idx} loss: {loss:.6f}')

    noisy, clean = next(iter(loader))
    noisy = noisy.to(device)
    clean = clean.to(device)
    with torch.no_grad():
        out = modelA(noisy, noisy)
    psnr = compute_psnr(out, clean)
    print(f'PSNR after ModelA demo run: {psnr:.2f}')

    print('Running ModelB forward pass on one batch...')
    modelB = ModelB(num_stages=3, num_filters=8, kernel_size=5, gamma=0.5).to(device)
    with torch.no_grad():
        out_b = modelB(noisy, noisy)
    psnr_b = compute_psnr(out_b, clean)
    print(f'PSNR ModelB warm-start output: {psnr_b:.2f}')


if __name__ == '__main__':
    main()
