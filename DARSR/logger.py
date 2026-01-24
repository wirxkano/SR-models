import logging


class Logger:
    def __init__(self, name: str = "app_logger", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        handlers = [logging.FileHandler(f"{name}.log", mode="a"), logging.StreamHandler()]
        for handler in handlers:
            handler.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_training_info(
        self, epoch: int, total_epochs: int, learning_rate: float
    ) -> None:
        return self.logger.info(
            f"Epoch [{epoch}/{total_epochs}] - Learning Rate: {learning_rate:.6f}"
        )

    def get_logger(self) -> logging.Logger:
        return self.logger


# Example usage:
if __name__ == "__main__":
    logger = Logger()
    app_logger = logger.get_logger()
    app_logger.info("This is an info message")
    app_logger.error("This is an error message")
    logger.log_training_info(epoch=1, total_epochs=10, learning_rate=0.001)
