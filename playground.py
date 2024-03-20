import cv2
import utilities

full_hd_path = r'./data/gt/full_hd/'
test = r'./data/'

gt_    = cv2.imread(f"{full_hd_path}1.png")
gt_tensor = utilities.convert_image(gt_, 'torch')
gt_ = utilities.convert_image(gt_tensor, 'cv2')
cv2.imwrite(test + 'test.png', gt_)