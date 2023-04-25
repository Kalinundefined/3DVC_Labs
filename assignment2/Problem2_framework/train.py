import torch
from torch.utils.data import DataLoader
from dataset import CubeDataset
from model import Img2PcdModel
from loss import CDLoss, HDLoss
from tqdm import tqdm
from statistics import mean
from pytorch3d.io import ply_io

def main():

    # TODO: Design the main function, including data preparation, training and evaluation processes.

    # Environment:
    device= 'cuda'
    # torch.manual_seed(42)
    # Directories:
    cube_data_path= '/home/jialin/repo22/3dvc/3dvc_1/assignment2/Problem2/cube_dataset/clean'
    output_dir= '/home/jialin/repo22/3dvc/3dvc_1/assignment2/Problem2/output'

    # Training hyper-parameters:
    batch_size= 64
    epoch= 15
    learning_rate= 1e-5

    # Data lists:
    training_cube_list= [i for i in range(100)][:80]
    test_cube_list= [i for i in range(100)][80:]
    view_idx_list= [i for i in range(16)]

    # Preperation of datasets and dataloaders:
    # Example:
    training_dataset = CubeDataset(cube_data_path, training_cube_list, view_idx_list, device=device)
    test_dataset = CubeDataset(cube_data_path, test_cube_list, view_idx_list, device=device)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Network:
    # Example:
    model = Img2PcdModel(device=device)

    # Loss:
    # Example:
    loss_fn = HDLoss()

    # Optimizer:
    # Example:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training process:
    # Example:
    print("==== train ====")
    model.train()
    for epoch_idx in tqdm(range(epoch)):
        epoch_loss = []
        for batch_idx, (data_img, data_pcd) in enumerate(training_dataloader):
            # forward
            pred = model(data_img)
            # compute loss
            loss = loss_fn(pred, data_pcd)
            with torch.no_grad():
                epoch_loss.append(loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tqdm.write(f"epoch {epoch_idx}, loss={mean(epoch_loss)}")
        
    # Final evaluation process:
    # Example:
    print("==== eval ====")
    model.eval()
    test_loss = []
    for batch_idx, (data_img, data_pcd) in enumerate(test_dataloader):

        # forward
        pred = model(data_img)
        if batch_idx == 1:
            ply_io.save_ply(f"/home/jialin/repo22/3dvc/3dvc_1/assignment2/Problem2_framework/output/output_pointcloud{batch_idx}_{1}.ply", pred[0])
        # compute loss
        loss = loss_fn(pred, data_pcd)
        test_loss.append(loss.item())
    print(f'test loss={mean(test_loss)}')
    


if __name__ == "__main__":
    main()
