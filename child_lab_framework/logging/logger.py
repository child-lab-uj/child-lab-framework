class Logger:

    @staticmethod
    def info(*args, **kwargs):
        print("[INFO]: ", *args, **kwargs)
