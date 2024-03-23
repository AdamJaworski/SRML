import pathlib
import numpy.random
from time import time
from model import Model
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from piqa import SSIM
import os
from utilities import convert_image

full_hd_path = r'./data/gt/full_hd/'
low_res_path = r'./data/lr/'
output_path  = r'./data/out/'
model_state   = r'./data/'

run_id = 'Beta'


def train_model(model_, loss_function, optimizer, data_list, gt_list) -> None:

    if not pathlib.Path.exists(pathlib.Path(output_path + run_id)):
        pathlib.Path(output_path + run_id).mkdir()

    epoch = 0
    while True:
        epoch += 1
        total_loss = 0.0
        errors = 0
        numpy.random.shuffle(data_list)
        for index, photo in enumerate(data_list):
            start = time()

            input_ = cv2.imread(f"{low_res_path}{photo}")
            gt_    = cv2.imread(f"{full_hd_path}{gt_list[gt_list.index(photo)]}")

            input_tensor = convert_image(input_, 'torch')
            gt_tensor    = convert_image(gt_, 'torch')

            running_loss = 0.0
            optimizer.zero_grad()

            setup = time()
            output_tensor = model_(input_tensor)
            nn_output = time()

            try:
                loss = 1 - loss_function(output_tensor, gt_tensor)
            except Exception as e:
                print('#' * (len(str(f'#### Error {e} with image: {photo} ####'))))
                print(f'#### Error {e} with image: {photo} ####')
                print('#' * (len(str(f'#### Error {e} with image: {photo} ####'))))
                errors += 1
                continue

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_loss += running_loss
            end = time()

            get_statistics(epoch, index, len(data_list), photo, running_loss, end, nn_output, setup, start)
            output_tensor = output_tensor.detach()
            output_cv2 = convert_image(output_tensor, 'cv2')
            cv2.imwrite(output_path + run_id + "/" + str(epoch) + "_" + photo, output_cv2)

            if pathlib.Path.exists(pathlib.Path(model_state + 'model_' + run_id + '.pth')):
                pathlib.Path.unlink(pathlib.Path(model_state + 'model_' + run_id + '.pth'))

            torch.save(model_.state_dict(), model_state + 'model_' + run_id + '.pth')
        end_of_epoch_summary(total_loss, errors, len(data_list), epoch)


def get_statistics(epoch, index, len_, photo, running_loss, end, nn_output, setup, start) -> None:
    print(f"epoch: {epoch:2}, finished: {index + 1:3}/{len_}, image: {photo:7}, loss: {running_loss:4.3}, "
          f"similarity: {(1 - round(running_loss, 3)) * 100}%, ttb: {end - nn_output :4.4}s, ttp: {nn_output - setup:4.4}s, "
          f"total time: {end - start:4.4}s")


def end_of_epoch_summary(total_loss, errors, total_images, epoch) -> None:
    epoch_report = open(output_path + run_id + '.txt', 'a+')
    summary_message = f"Finished {epoch} epoch: Processed {total_images} total images, with {errors} errors. " \
                      f"Avg loss was: {total_loss/(total_images - errors):4.4}, Avg"
    epoch_report.write(summary_message + '\n')


if __name__ == "__main__":
    model__ = Model()
    model__ = torch.jit.script(model__)
    if pathlib.Path.exists(pathlib.Path(model_state + 'model.pth')):
        model__.load_state_dict(torch.load(model_state + 'model.pth'))
    loss_function_ = SSIM()
    train_model(model__, loss_function_, optim.Adam(model__.parameters(), lr=1e-4), os.listdir(low_res_path), os.listdir(full_hd_path))

