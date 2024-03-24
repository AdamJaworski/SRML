import pathlib
import rawpy
import os
import math
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2


full_hd_path = r'./data/gt/full_hd/'
full_hd_bicubic_path = r'./data/gt/full_hd_bicubic/'
high_res_path = r'./data/gt/high_res/'
low_res_path = r'./data/lr/'


def create_full_hd() -> bool:
    high_res_files = os.listdir(high_res_path)
    for index, file in enumerate(high_res_files):
        img = cv2.imread(high_res_path + file)

        try:
            if img is None:
                raw = rawpy.imread(high_res_path + file)
                img = raw.postprocess()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if img is None:
                    print(f"{file} couldn't be read")
                    continue
        except Exception as e:
            print(high_res_path + file)
            continue

        if img.shape[1] >= img.shape[0]:
            size2 = int(math.ceil(img.shape[0] / (img.shape[1] / 1920)))
            if size2 % 2 == 1:
                size2 += 1
            size = (1920, size2)

        else:
            size = (int(math.ceil(img.shape[1] / (img.shape[0] / 1920))), 1920)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)

        i = 1
        while pathlib.Path.exists(pathlib.Path(f"{full_hd_path}{i}.png")):
            i += 1
        cv2.imwrite(f"{full_hd_path}{i}.png", img)
        print(f"Resized photos:  {index}/{len(high_res_files)}")

    return True


def create_full_hd_upscale():
    low_res_files = os.listdir(low_res_path)
    for index, file in enumerate(low_res_files):
        img = cv2.imread(low_res_path + file)
        size1 = int(math.ceil(img.shape[1] * 2))
        size2 = int(math.ceil(img.shape[0] * 2))
        img = cv2.resize(img, (size1, size2), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(full_hd_bicubic_path + file, img)
        print(f"Resized photos:  {index}/{len(low_res_files)}")


def create_low_res() -> bool:
    full_hd_files = os.listdir(full_hd_path)
    for index, file in enumerate(full_hd_files):
        img = cv2.imread(full_hd_path + file)

        if img is None:
            raw = rawpy.imread(high_res_path + file)
            img = raw.postprocess()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if img is None:
                print(f"{file} couldn't be read")
                continue
        size1 = int(math.ceil(img.shape[1] / 2))
        size2 = int(math.ceil(img.shape[0] / 2))
        img = cv2.resize(img, (size1, size2), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(f"{low_res_path}{file}", img)
        print(f"Resized photos:  {index}/{len(full_hd_files)}")

    return True


def rename_high():
    high_res = r'./data/gt/high_res/'
    files = os.listdir(high_res)
    i = 0
    for file in files:
        while pathlib.Path.exists(pathlib.Path(high_res + str(i) + '.png')):
            i += 1
        os.rename(high_res + file, high_res + str(i) + '.png')
        i += 1


if __name__ == "__main__":
    create_full_hd()
    create_low_res()
    create_full_hd_upscale()
