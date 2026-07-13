"""Validate and prepare vacancy data."""

import logging

from joblens.data.cleaning import prepare_vacancies
from joblens.data.labeling import add_role_target
from joblens.data.loading import load_vacancies, save_vacancies
from joblens.settings import load_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Prepare configured raw data, falling back to the explicit sample path."""
    settings = load_settings()
    config = settings.values
    source = settings.path("raw_data")
    if not source.exists():
        source = settings.path("sample_data")
        LOGGER.warning("Raw dataset absent; using synthetic smoke-test dataset: %s", source)
    frame = load_vacancies(source)
    clean = prepare_vacancies(frame, config["salary"]["min"], config["salary"]["max"])
    clean = add_role_target(clean)
    save_vacancies(clean.drop(columns=["key_skills_list"]), settings.path("processed_data"))
    LOGGER.info("Prepared %d rows into %s", len(clean), settings.path("processed_data"))


if __name__ == "__main__":
    main()
