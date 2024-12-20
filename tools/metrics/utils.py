class AverageMeter:
    """
    Computes and stores the average and current value.

    Methods
    -------
    __init__():
        Initializes the AverageMeter and resets its values.
    reset():
        Resets all the values (val, avg, sum, count) to zero.
    update(val, n=1):
        Updates the meter with a new value.

        Parameters
        ----------
        val : float
            The new value to be added.
        n : int, optional
            The weight of the new value (default is 1).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


