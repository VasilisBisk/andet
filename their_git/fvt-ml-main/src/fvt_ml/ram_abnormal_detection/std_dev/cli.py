import logging
from typing import Optional

import typer

from fvt_ml.ram_abnormal_detection.std_dev.app import StDevApp
from fvt_ml.ram_abnormal_detection.std_dev.config import Config

LOG_FORMAT = "%(levelname)s:     %(asctime)-15s - %(filename)s: %(message)s"

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command(name="infer")
def infer(start_date: str, end_date: str, target_id: str) -> None:

    conf = Config(target_id=target_id, start_date=start_date, end_date=end_date)

    logging.basicConfig(level=getattr(logging, conf.log_level, "INFO"), format=LOG_FORMAT)

    logger.info(
        "Starting inference job for TARGET_ID: %s START_DATE: %s END_DATE: %s",
        target_id,
        start_date,
        end_date,
    )

    conf = Config(target_id=target_id, start_date=start_date, end_date=end_date)

    model = conf.get_model()
    preprocessor = conf.get_preprocessor()
    postprocessor = conf.get_postprocessor()
    raw_data_repo = conf.get_raw_data_repo()
    predictions_repo = conf.get_predictions_repo()

    app = StDevApp(
        model=model,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        raw_data_repo=raw_data_repo,
        predictions_repo=predictions_repo,
    )
    app.infer(start_date=conf.start_date, end_date=conf.end_date, target_id=conf.target_id)

    logger.info("Inference finished successfully!")


@app.command(name="train")
def train(
    start_date: str,
    end_date: str,
    target_id: str,
    output_path: Optional[str] = None,
) -> None:

    conf = Config(target_id=target_id, start_date=start_date, end_date=end_date)

    logging.basicConfig(level=getattr(logging, conf.log_level, "INFO"), format=LOG_FORMAT)

    logger.info(
        "Starting training job for TARGET_ID: %s START_DATE: %s END_DATE: %s",
        target_id,
        start_date,
        end_date,
    )

    if output_path:
        if not output_path.endswith(".json"):
            raise Exception("Make sure to store your data as JSON file")

    preprocessor = conf.get_preprocessor()
    raw_data_repo = conf.get_raw_data_repo()

    app = StDevApp(preprocessor=preprocessor, raw_data_repo=raw_data_repo)
    app.train(
        start_date=conf.start_date,
        end_date=conf.end_date,
        target_id=conf.target_id,
        output_path=output_path,
    )

    logger.info("Training finished successfully!")
