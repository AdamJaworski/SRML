import pathlib
from datetime import datetime
import numpy.random
from time import time
from model import Model
import cv2
import torch
from path_manager import PathManager
import torch.optim as optim
from piqa import SSIM
import os
from utilities import convert_image
from train_options import opt

full_hd_path = r'./data/gt/full_hd/'
low_res_path = r'./data/lr/'
output_path  = r'./data/out/'

model_path_manager = None


def train_model(model_, loss_function, optimizer, data_list, gt_list) -> None:
    epoch             = opt.STARTING_EPOCH
    processed_images  = 0
    processed_images_ = 0
    highest_loss      = 0
    lowest_loss       = 1
    total_time        = 0.0

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

            output_tensor = model_(input_tensor)

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

            if running_loss < lowest_loss:
                lowest_loss = running_loss
            if running_loss > highest_loss:
                highest_loss = running_loss

            processed_images  += 1
            processed_images_ += 1
            total_time += end - start

            if processed_images % opt.SAVE_MODEL_AFTER == 0:
                save_model(model_, f'{processed_images}_{epoch}')

            if processed_images % opt.PRINT_RESULTS == 0:
                save_output(output_tensor, epoch, processed_images, photo)
                get_stats(epoch, total_loss, errors, processed_images_, total_time, lowest_loss, highest_loss)
                total_loss, errors, processed_images_, total_time, lowest_loss, highest_loss = 0.0, 0, 0, 0.0, 1, 0


def save_output(output_tensor, epoch, processed_images, name):
    output_tensor = output_tensor.detach()
    output_cv2 = convert_image(output_tensor, 'cv2')
    image = str(epoch) + '_' + str(processed_images) + '_' + name
    cv2.imwrite(str(model_path_manager.out_path) + '/' + image, output_cv2)


def save_model(model_instance, name_of_save: str):
    # Current save
    if pathlib.Path.exists(model_path_manager.root_path / (name_of_save + '.pth')):
        pathlib.Path.unlink(model_path_manager.root_path / (name_of_save + '.pth'))
    torch.save(model_instance.state_dict(), model_path_manager.root_path / (name_of_save + '.pth'))

    # Latest save
    if pathlib.Path.exists(model_path_manager.root_path / 'latest.pth'):
        pathlib.Path.unlink(model_path_manager.root_path / 'latest.pth')
    torch.save(model_instance.state_dict(), model_path_manager.root_path / 'latest.pth')


def get_stats(epoch, total_loss, errors, total_images, total_time, lowest_loss, highest_loss) -> None:
    loss_file = open(model_path_manager.loss_file, 'a+')
    summary_message = f"(epoch: {epoch}, iters: {total_images}, errors: {errors}, time: {total_time:.3}s) " \
                      f"a_loss: {total_loss/total_images:.3}, a_SSIM: {round(1 - total_loss/total_images, 3) * 100:.3f}%, " \
                      f"a_time: {total_time/total_images:.3}, b_SSIM: {round(1 - lowest_loss, 3) * 100:.3f}%, " \
                      f"l_SSIM: {round(1 - highest_loss, 3) * 100:.3f}%"
    print(summary_message)
    loss_file.write(summary_message + '\n')
    loss_file.close()


if __name__ == "__main__":
    # torch.set_num_threads(1)
    model = Model()
    loss_function_ = SSIM()
    model_path_manager = PathManager(opt.MODEL)
    model = torch.jit.script(model)

    if opt.CONTINUE_LEARNING:
        model.load_state_dict(torch.load(model_path_manager.root_path / 'latest.pth'))

    loss_file = open(model_path_manager.loss_file, 'a+')
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    start_message = '='*50 + f'{dt_string}' + '='*50
    loss_file.write(start_message + '\n')
    loss_file.close()

    print(f"Starting training with rate: {opt.LR}")
    train_model(model, loss_function_, optim.Adam(model.parameters(), lr=opt.LR), os.listdir(low_res_path), os.listdir(full_hd_path))

