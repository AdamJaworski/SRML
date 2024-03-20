import pathlib
from model import Model
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import os
from utilities import convert_image

full_hd_path = r'./data/gt/full_hd/'
low_res_path = r'./data/lr/'
output_path  = r'./data/out/'
model_state   = r'./data/model.pth'

run_id = 'gamma2'


def train_model(model_: torch.nn.Module, loss_function) -> None:
    data_list = os.listdir(low_res_path)
    gt_list   = os.listdir(full_hd_path)

    optimizer = optim.Adam(model_.parameters(), lr=1e-4)

    if not pathlib.Path.exists(pathlib.Path(output_path + run_id)):
        pathlib.Path(output_path + run_id).mkdir()

    epoch = 0
    while True:
        epoch += 1
        for index in range(len(data_list)):
            if gt_list[index] != data_list[index]:
                print(f"Error: {gt_list[index]}  {data_list[index]}")

            input_ = cv2.imread(f"{low_res_path}{data_list[index]}")
            gt_    = cv2.imread(f"{full_hd_path}{gt_list[index]}")

            input_tensor = convert_image(input_, 'torch')
            gt_tensor    = convert_image(gt_, 'torch')

            running_loss = 0.0
            optimizer.zero_grad()
            output_tensor = model_(input_tensor)
            try:
                loss = loss_function(output_tensor, gt_tensor)
            except Exception as e:
                print('#' * 30)
                print(f'Error {e} with image: {data_list[index]}')
                print('#' * 30)
                continue
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"epoch: {epoch:2}, finished: {index + 1:3}/{len(data_list)}, image: {data_list[index]:7}, loss: {running_loss:.3}")

            output_tensor = output_tensor.detach()
            output_cv2 = convert_image(output_tensor, 'cv2')
            cv2.imwrite(output_path + run_id + "/" + str(epoch) + "_" + data_list[index], output_cv2)

            if pathlib.Path.exists(pathlib.Path(model_state)):
                pathlib.Path.unlink(pathlib.Path(model_state))

            torch.save(model_.state_dict(), model_state)


if __name__ == "__main__":
    model__ = Model()
    if pathlib.Path.exists(pathlib.Path(model_state)):
        model__.load_state_dict(torch.load(model_state))
    loss_function_ = nn.MSELoss()
    train_model(model__, loss_function_)

