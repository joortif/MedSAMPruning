

class NoValidAutobatchConfigException(Exception):
    def __init__(self, usable_mem, message=None):
        if message is None:
            message = (
                f"No valid configuration could fit within the usable GPU memory limit "
                f"({usable_mem / (1 << 30):.2f} GB). "
                "Try freeing up GPU memory, reducing model size, or increasing the usable memory fraction."
            )
        super().__init__(message)
        self.usable_mem = usable_mem