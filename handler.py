
import logging

from sentiment_model import SentimentModel


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

model = SentimentModel()


def handle(event, context):
    if event.get("source") == "KEEP_LAMBDA_WARM":
        LOGGER.info("No ML work to do. Just staying warm...")
        return "Keeping Lambda warm"

    return {
        "sentiment": model.predict(text=event["text"])
    }
