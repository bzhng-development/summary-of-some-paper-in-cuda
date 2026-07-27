"""Inclusive publication-date windows shared by paper discovery sources."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from pydantic import BaseModel, ConfigDict, model_validator


class DateWindow(BaseModel):
    """A validated inclusive calendar-date boundary."""

    model_config = ConfigDict(frozen=True)

    since: date
    through: date

    @model_validator(mode="after")
    def validate_order(self) -> DateWindow:
        if self.through < self.since:
            raise ValueError("through must be on or after since")
        return self

    @classmethod
    def from_inputs(
        cls,
        *,
        years: str | None = None,
        since: str | None = None,
        through: str | None = None,
        days: int | None = None,
        today: date | None = None,
        default_years: str = "2025-2026",
    ) -> DateWindow:
        """Resolve either a year window or an exact elapsed-day delta."""
        exact_requested = since is not None or through is not None or days is not None
        if exact_requested and years is not None:
            raise ValueError("--window cannot be combined with --days, --since, or --through")
        if since is not None and days is not None:
            raise ValueError("--since and --days are alternatives")
        if days is not None and days < 0:
            raise ValueError("--days must be non-negative")

        if exact_requested:
            end = _parse_date(through) if through else today or datetime.now().astimezone().date()
            start = _parse_date(since) if since else end - timedelta(days=days or 0)
            return cls(since=start, through=end)

        year_start, year_end = _parse_years(years or default_years)
        return cls(since=date(year_start, 1, 1), through=date(year_end, 12, 31))

    @property
    def years(self) -> str:
        """Return the smallest whole-year query that contains this window."""
        if self.since.year == self.through.year:
            return str(self.since.year)
        return f"{self.since.year}-{self.through.year}"

    @property
    def label(self) -> str:
        """Filesystem-safe exact window label."""
        return f"{self.since.isoformat()}_{self.through.isoformat()}"

    def contains(self, raw_date: str | None) -> bool:
        """Return whether an ISO date or datetime falls inside the window."""
        parsed = parse_optional_date(raw_date)
        return parsed is not None and self.since <= parsed <= self.through


def parse_optional_date(raw_date: str | None) -> date | None:
    """Parse an ISO date or datetime without raising on absent metadata."""
    if not raw_date:
        return None
    try:
        return date.fromisoformat(raw_date.strip()[:10])
    except ValueError:
        return None


def validate_openalex_save_mode(*, save_neon: bool, exact_window: bool) -> None:
    """Prevent exact-date OpenAlex rows from bypassing authoritative arXiv dates."""
    if save_neon and exact_window:
        raise ValueError(
            "Exact-date OpenAlex runs cannot save directly because OpenAlex publication dates "
            "can differ from initial arXiv dates. Use paper-e2e, or pass --no-save-neon and "
            "absorb the JSONL through authoritative arXiv date verification."
        )


def _parse_date(raw_date: str) -> date:
    try:
        return date.fromisoformat(raw_date)
    except ValueError as error:
        raise ValueError(f"Expected YYYY-MM-DD, got {raw_date!r}") from error


def _parse_years(raw_years: str) -> tuple[int, int]:
    parts = raw_years.split("-", maxsplit=1)
    try:
        start = int(parts[0])
        end = int(parts[-1])
    except ValueError as error:
        raise ValueError(f"Expected YYYY or YYYY-YYYY, got {raw_years!r}") from error
    if start < 1991 or end < start:
        raise ValueError(f"Invalid year window: {raw_years!r}")
    return start, end
