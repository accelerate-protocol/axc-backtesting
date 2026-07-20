import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import datetime
import requests
from icecream import ic

__all__ = ['TokenPriceProcessor']

ic.disable()

class TokenPriceProcessor:
    """A class to fetch, process, and upload token price data for a given vault."""

    def __init__(self, base_url: str, token: str, load_url: str):
        """Initializes the processor with configuration values.

        Args:
            base_url: Base URL for asset metrics API endpoint.
            token: Token identifier for price metrics.
            load_url: URL for fetching token prices from vault.
        """
        self.base_url = base_url
        self.token = token
        self.load_url = load_url

    def fetch_token_prices(self, vault_id: str) -> dict:
        """
        Fetches token price data for the given vault ID from the specified endpoint.

        Args:
            vault_id: The ID of the vault to fetch prices for.

        Returns:
            dict: Raw JSON response containing price data.

        Raises:
            RuntimeError: If the HTTP request fails or JSON decoding fails.
        """
        url = f"{self.load_url}?vaultId={vault_id}"
        req = Request(url, headers={"Accept": "application/json"})

        try:
            with urlopen(req, timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {response.reason}")
                data = response.read().decode("utf-8")
        except HTTPError as e:
            raise RuntimeError(f"HTTP error {e.code}: {e.reason}") from e
        except URLError as e:
            raise RuntimeError(f"Network error: {e.reason}") from e

        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON received: {e}") from e

    @staticmethod
    def unix_to_date(unix_ts: int) -> str:
        """
        Converts a Unix timestamp to a formatted date string (YYYY-MM-DD).

        Args:
            unix_ts: Unix timestamp to convert.

        Returns:
            str: Formatted date string in YYYY-MM-DD format.
        """
        dt = datetime.datetime.fromtimestamp(unix_ts, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d")

    def save_prices(
        self,
        prices: list[dict],
        api_key: str,
        dryrun: bool = True,
        start_date: str | None = None
    ) -> None:
        """Fetch and upload token price metrics.
        
        When start_date is None, processes only the most recent price entry.
        Filters dates upfront for efficiency.
        """
        if not prices:
            return

        # Compute effective start_date
        if start_date is None:
            start_date = self.unix_to_date(prices[-1]["pricedAt"])
        ic(start_date)
        # Build filtered list of (entry, date_str) tuples in one pass
        filtered_entries = [
            (entry, self.unix_to_date(entry["pricedAt"]))
            for entry in prices
            if self.unix_to_date(entry["pricedAt"]) >= start_date
        ]
        ic(filtered_entries)
        # Process filtered entries
        for entry, date_str in filtered_entries:
            url = f"{self.base_url}/{date_str}"
            payload = [{
                "id": self.token,
                "metrics": {"net_asset_value": float(entry["price"])}
            }]

            print(f"date={date_str} token={self.token} price={entry['price']}")

            if not dryrun:
                response = requests.put(
                    url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload
                )
                print(f"Posting to {url}\n  Status: {response.status_code}")
                if response.text:
                    print(f"  Body: {response.text[:500]}")

    def process_prices(self, vault: str, api_key: str, dryrun: bool = True, start_date: str = None) -> None:
        """
        Main method to fetch and process token prices for a given vault.

        Args:
            vault: The ID of the vault to fetch prices for.
            api_key: Bearer token for API authentication.
            dryrun: If True, simulates the upload without making actual requests.
            start_date: Optional start date (YYYY-MM-DD) to filter entries.
        """
        prices = self.fetch_token_prices(vault)['data']['list']
        self.save_prices(prices, api_key, dryrun, start_date)