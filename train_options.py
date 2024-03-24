class Options:
    def __init__(self):
        pass


opt = Options()


def add_argument(**kwargs):
    for key, value in kwargs.items():
        setattr(opt, key, value)


add_argument(SAVE_MODEL_AFTER=20)
add_argument(PRINT_RESULTS=20)
add_argument(CONTINUE_LEARNING=True)
add_argument(MODEL='Alpha')
add_argument(LR=2e-5)
add_argument(STARTING_EPOCH=0)
