import pathlib
import rawpy
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2


full_hd_path = r'./data/gt/full_hd/'
high_res_path = r'./data/gt/high_res/'
low_res_path = r'./data/lr/'


def create_full_hd() -> bool:
    high_res_files = os.listdir(high_res_path)
    for index, file in enumerate(high_res_files):
        img = cv2.imread(high_res_path + file)

        if img is None:
            raw = rawpy.imread(high_res_path + file)
            img = raw.postprocess()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if img is None:
                print(f"{file} couldn't be read")
                continue

        if img.shape[1] >= img.shape[0]:
            size = (1920, int(img.shape[0] / (img.shape[1] / 1920)))
        else:
            size = (int(img.shape[1] / (img.shape[0] / 1920)), 1920)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)

        i = 1
        while pathlib.Path.exists(pathlib.Path(f"{full_hd_path}{i}.png")):
            i += 1
        cv2.imwrite(f"{full_hd_path}{i}.png", img)
        print(f"Resized photos:  {index}/{len(high_res_files)}")

    return True


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

        # TODO Round up
        img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(f"{low_res_path}{file}", img)
        print(f"Resized photos:  {index}/{len(full_hd_files)}")

    return True


if __name__ == "__main__":
    create_low_res()
