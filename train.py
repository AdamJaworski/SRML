import pathlib

import numpy.random
from time import time
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

run_id = 'Theta'


def train_model(model_, loss_function, optimizer, data_list, gt_list) -> None:

    if not pathlib.Path.exists(pathlib.Path(output_path + run_id)):
        pathlib.Path(output_path + run_id).mkdir()

    epoch = 0
    while True:
        epoch += 1
        numpy.random.shuffle(data_list)
        for index, photo in enumerate(data_list):
            start = time()

            gt_index = gt_list.index(photo)
            if gt_list[gt_index] != photo:
                print(f"Error: {gt_list[gt_index]}  {photo}")

            input_ = cv2.imread(f"{low_res_path}{photo}")
            gt_    = cv2.imread(f"{full_hd_path}{gt_list[gt_index]}")

            input_tensor = convert_image(input_, 'torch')
            gt_tensor    = convert_image(gt_, 'torch')

            running_loss = 0.0
            optimizer.zero_grad()

            setup = time()
            output_tensor = model_(input_tensor)

            # Changing tensor from [-1,1] to [0,1]
            output_tensor = (output_tensor + 1) / 2

            nn_output = time()
            try:
                loss = loss_function(output_tensor, gt_tensor)
            except Exception as e:
                print('#' * (len(str(f'#### Error {e} with image: {photo} ####'))))
                print(f'#### Error {e} with image: {photo} ####')
                print('#' * (len(str(f'#### Error {e} with image: {photo} ####'))))
                continue

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            end = time()

            get_statistics(epoch, index, len(data_list), photo, running_loss, end, nn_output, setup, start)
            output_tensor = output_tensor.detach()
            output_cv2 = convert_image(output_tensor, 'cv2')
            cv2.imwrite(output_path + run_id + "/" + str(epoch) + "_" + photo, output_cv2)

            if pathlib.Path.exists(pathlib.Path(model_state)):
                pathlib.Path.unlink(pathlib.Path(model_state))

            torch.save(model_.state_dict(), model_state)


def get_statistics(epoch, index, len_, photo, running_loss, end, nn_output, setup, start) -> None:
    print(
        f"epoch: {epoch:2}, finished: {index + 1:3}/{len_}, image: {photo:7}, loss: {running_loss:.3}, ttb: {end - nn_output :.4}s, ttp: {nn_output - setup:.4}s, total time: {end - start:.4}s")


if __name__ == "__main__":
    model__ = Model()
    scripted_model = torch.jit.script(model__)
    if pathlib.Path.exists(pathlib.Path(model_state)):
        model__.load_state_dict(torch.load(model_state))
    loss_function_ = nn.MSELoss()
    train_model(scripted_model, loss_function_, optim.Adam(scripted_model.parameters(), lr=1e-4), os.listdir(low_res_path), os.listdir(full_hd_path))

