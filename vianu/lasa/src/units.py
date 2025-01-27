from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
from pathlib import Path
import re
from typing import List, Self, ClassVar

import pandas as pd

from vianu.lasa.settings import SMC_NON_DRUGS, SOURCES, SMC_FILENAME, FDA_FILENAME

logger = logging.getLogger(__name__)


@dataclass
class Product:
    """Medicinal product."""

    name: str | None = None
    license_holder: str | None = None
    valid_until: datetime | None = None
    active_substance: str | None = None


@dataclass
class AuthorizationUnit(ABC):
    """Objects for authorized medicines at a given source."""

    source: str | None = None
    products: List[Product] = field(default_factory=list)

    @classmethod
    @abstractmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        """Create an instance from a DataFrame."""
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, filename: Path) -> Self:
        """Create an instance from a file."""
        pass

    def get_products_as_df(self) -> pd.DataFrame:
        """Return the products as a DataFrame."""
        return pd.DataFrame([asdict(p) for p in self.products])


_SMC_COLUMNS = [
    "authorization_id",
    "dose_strength_id",
    "description",
    "license_holder",
    "medicinal_product_cat",
    "submission_dose_strength_cat",
    "submission_medicinal_product_cat",
    "first_authorization_date",
    "authorization_date",
    "valid_until",
]


@dataclass
class SwissmedicAuthorization(AuthorizationUnit):
    """Objects for authorized medicines at Swissmedic."""

    _source_url: ClassVar[str] = (
        "https://www.swissmedic.ch/swissmedic/en/home/services/listen_neu.html"
    )

    _skiprows: ClassVar[int] = 7
    _columns: ClassVar[List[str]] = _SMC_COLUMNS
    _format: ClassVar[str] = "%m/%d/%Y"

    @staticmethod
    def _extract_drug_name(text: str) -> str:
        """Extract the drug name from the description in the table."""
        # The cell content can be comma separated as: drug info, route of admission
        drug_info = text.split(",")[0]

        # Remove the numbers from the drug info (numbers usually hint to the dose and follow the drug name)
        drug_info = re.sub(r" [0-9]*", " ", drug_info)
        drug_info = re.sub(r"/[0-9]*", " ", drug_info)

        # Remove some common non-drug names
        words = [w for w in drug_info.split() if w.lower() not in SMC_NON_DRUGS]
        drug_name = " ".join(words)

        return drug_name.strip().strip("-").strip()

    @classmethod
    def _from_df(cls, df: pd.DataFrame) -> Self:
        products = []
        for _, row in df.iterrows():
            valid_until = row["valid_until"]
            product = Product(
                name=row["drug_name"],
                license_holder=row["license_holder"],
                valid_until=valid_until if valid_until != "unbegrenzt" else None,
            )
            products.append(product)
        return cls(source="swissmedic", products=products)

    @classmethod
    def _from_xlsx(cls, filename: Path) -> Self:
        """Create an instance from an Excel file."""
        df = pd.read_excel(
            filename,
            skiprows=7,
            header=None,
            names=cls._columns,
            date_format=cls._format,
        )
        return cls.from_df(df)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        """Create an instance from a DataFrame."""
        df = df.copy().drop_duplicates(subset=["description"])
        df["drug_name"] = df["description"].apply(cls._extract_drug_name)
        df = df.drop_duplicates(subset=["drug_name"])
        df = df.dropna(subset=["drug_name"])
        df.reset_index(inplace=True, drop=True)
        return cls._from_df(df=df)

    @classmethod
    def from_file(cls, filename: Path) -> Self:
        """Create an instance from a file."""
        return cls._from_xlsx(filename=filename)


class FDAAuthorization(AuthorizationUnit):
    """Objects for authorized medicines at the FDA."""

    _source_url: ClassVar[str] = (
        "https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files"
    )

    @staticmethod
    def _on_bad_lines(line: str) -> None:
        logger.warning(f"Skipping line: {line}")
        return None  # Skip the line

    @classmethod
    def _from_df(cls, df: pd.DataFrame) -> Self:
        products = []
        for _, row in df.iterrows():
            product = Product(
                name=row["DrugName"],
                active_substance=row["ActiveIngredient"],
            )
            products.append(product)
        return cls(source="fda", products=products)

    @classmethod
    def _from_csv(cls, filename: Path) -> Self:
        """Create an instance from a CSV file."""
        df = pd.read_csv(
            filename,
            sep="\t",
            header=0,
            on_bad_lines=cls._on_bad_lines,
            engine="python",
        )
        return cls.from_df(df)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        """Create an instance from a DataFrame."""
        df = df.copy().drop_duplicates(subset=["DrugName"])
        df = df.dropna(subset=["DrugName"])
        df.reset_index(inplace=True, drop=True)
        return cls._from_df(df=df)

    @classmethod
    def from_file(cls, filename: Path) -> Self:
        """Create an instance from a file."""
        return cls._from_csv(filename=filename)


_FILES = [SMC_FILENAME, FDA_FILENAME]
_AUTHS = [SwissmedicAuthorization, FDAAuthorization]
if not len(_FILES) == len(_AUTHS) == len(SOURCES):
    raise ValueError("Number of files, authorizations, and sources must be equal.")


class AuthorizationFactory:
    """Factory for authorizations."""

    _FILE_PATH = Path(__file__).parents[1] / "files"
    _FILE_NAME_MAP = {src: fn for src, fn in zip(SOURCES, _FILES)}
    _AUTHORIZATIONS = {src: auth for src, auth in zip(SOURCES, _AUTHS)}

    @classmethod
    def create(cls, source: str) -> AuthorizationUnit:
        authorization = cls._AUTHORIZATIONS.get(source)
        if authorization is None:
            raise ValueError(f"Source={source} not supported.")
        return authorization.from_file(
            filename=cls._FILE_PATH / cls._FILE_NAME_MAP[source]
        )
