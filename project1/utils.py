class EarlyStopping:
    def __init__(self, patience, delta=0, mode='max'):
        self.patience = patience          # Number of training epochs we are willing to wait for to observe an improvement of the model
        self.delta = delta                # Quantifying the improvement we want to observe
        self.mode = mode                  # Defining the improvement: an increase (e.g. accuracy) or a decrease (e.g. loss)
        self.best_value = None            # The best value of the metric of interest observed so far
        self.counter = 0                  # The number of epochs we have already waited for

    def __call__(self, value):
        if self.patience is None:         # To disable Early stopping and keep on training for nb_epochs
            return False

        if self.best_value is None:       # If we have not yet observed any value
            self.best_value = value       # then take the first as best
            self.counter = 0              # start to count
            return False                  # and keep on training

        if self.mode == 'max':                          # We want to observe an increase in accuracy
            if value > self.best_value + self.delta:    # if the current value is bigger than the previous best value by an amount delta
                return self._positive_update(value)     
            else:
                return self._negative_update(value)

        elif self.mode == 'min':                        # We want to observe an decrease in loss
            if value < self.best_value - self.delta:    # if the current value is smaller than the previous best value by an amount delta
                return self._positive_update(value)
            else:
                return self._negative_update(value)

        else:
            raise ValueError(f"Illegal mode for early stopping: {self.mode}")

    # Start to count againg, update the best value and keep on training
    def _positive_update(self, value):
        self.counter = 0
        self.best_value = value

        return False

    # Keep on counting, our patience is running out
    def _negative_update(self, value):
        self.counter += 1
        if self.counter > self.patience:    # We have been patient for long enough,
            return True                     # further improvements are unlikely to occur: stop training
        else:
            return False                    
