# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from datetime import datetime
from urllib.parse import urlparse
from pathlib import PurePosixPath

# ---------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------
from google.cloud import storage
import requests


class GCS:
    """
    Google Cloud Storage client for uploading and downloading files.
    """

    def __init__(
        self,
        bucket_name: str
    ):
        """
        Initialize the GCS client.

        Args:
            bucket_name (str): The name of the GCS bucket.

        Returns:
            None
        """
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)


    @staticmethod
    def _normalize_path(path: str) -> str:
        """
        Normalize a GCS path to a POSIX path.

        Args:
            path (str): The GCS path to normalize.

        Returns:
            str: The normalized POSIX path.
        """

        if path.startswith("gs://"):
            parsed = urlparse(path)
            path = parsed.path.lstrip("/")

        return str(PurePosixPath(path))


    @staticmethod
    def _timestamp() -> str:
        """
        Get the current timestamp.

        Args:
            None

        Returns:
            str: The current timestamp.
        """

        return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


    def exists(self, gcs_path: str) -> bool:
        """
        Check if a blob exists in the bucket.

        Args:
            gcs_path (str): Path inside the bucket.

        Returns:
            bool: True if exists, False otherwise.
        """
        gcs_path = self._normalize_path(gcs_path)
        blob = self.bucket.blob(gcs_path)

        return blob.exists(self.client)


    def upload(
        self,
        local_path: str,
        gcs_path: str,
        name: str | None = None,
        timestamp: bool = False,
    ):
        """
        Upload a file to Google Cloud Storage.

        Args:
            local_path (str): The local file path to upload.
            gcs_path (str): The GCS path to upload to.
            name (str | None): The name to use for the uploaded file (optional).
            timestamp (bool): Whether to add a timestamp to the uploaded file name.

        Returns:
            uri (str): The GCS path where the file was uploaded.
        """

        gcs_path = self._normalize_path(gcs_path)

        if name:
            if timestamp:
                stem, *ext = name.rsplit(".", 1)
                suffix = f".{ext[0]}" if ext else ""
                name = f"{stem}_{self._timestamp()}{suffix}"

            gcs_path = str(PurePosixPath(gcs_path) / name)

        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)

        uri = f"gs://{blob.bucket.name}/{gcs_path}"

        # logger.info(f"Uploaded {local_path} → {uri}")
        return uri


    def upload_from_url(
        self,
        url: str,
        gcs_path: str,
        timeout: int = 10,
        name: str | None = None,
        headers: dict | None = None,
    ):
        """
        Upload a file from a URL to Google Cloud Storage.

        Args:
            url (str): The URL of the file to upload.
            gcs_path (str): The GCS path to upload to.
            name (str | None): The name to use for the uploaded file (optional).
            timestamp (bool): Whether to add a timestamp to the uploaded file name.

        Returns:
            uri (str): The GCS path where the file was uploaded.
        """

        headers = headers or {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        gcs_path = self._normalize_path(gcs_path)
        object_path = str(PurePosixPath(gcs_path) / name)

        blob = self.bucket.blob(object_path)
        blob.upload_from_string(
            response.content,
            content_type=response.headers.get("Content-Type", "image/jpeg"),
        )

        uri = f"gs://{blob.bucket.name}/{object_path}"
        # logger.info(f"Uploaded {url} → {uri}")

        return uri

    def download(self, gcs_path: str, local_path: str):
        """
        Download a file from Google Cloud Storage.

        Args:
            gcs_path (str): The GCS path to download.
            local_path (str): The local path to save the downloaded file.

        Returns:
            local_path(str): The local path where the file was downloaded.
        """

        gcs_path = self._normalize_path(gcs_path)
        self.bucket.blob(gcs_path).download_to_filename(local_path)

        # logger.info(f"Downloaded gs://{self.bucket.name}/{gcs_path} → {local_path}")
        return local_path


    def download_bytes(self, gcs_path: str) -> bytes:
        """
        Download file content into memory (for Gemini multimodal, etc.)

        Args:
            gcs_path (str): The GCS path to download.

        Returns:
            bytes: The content of the downloaded file.
        """

        gcs_path = self._normalize_path(gcs_path)
        return self.bucket.blob(gcs_path).download_as_bytes()


    def delete(self, gcs_path: str):
        """
        Delete a file from Google Cloud Storage.

        Args:
            gcs_path (str): The GCS path to delete.

        Returns:
            None
        """

        gcs_path = self._normalize_path(gcs_path)
        self.bucket.blob(gcs_path).delete()

        # logger.info(f"Deleted gs://{self.bucket.name}/{gcs_path}")


if __name__ == "__main__":

    # Upload a file
    gcs_client = GCS(bucket_name="crp-qas-dig-plantillasenr-bkt02")
    gcs_client.upload(
        local_path="src/mocks/mock.json",
        gcs_path="new_templates/templates_silver/",
        timestamp=True,
        name="testing_file.json"
    )

    # Upload a file from a URL
    gcs_client = GCS(bucket_name="crp-qas-dig-plantillasenr-bkt02")
    uri = gcs_client.upload_from_url(
        url="https://m.media-amazon.com/images/I/71-RcYdHFxL._AC_SY695_.jpg",
        gcs_path="historical_upload/templates/input_images",
        name="testing_image.jpg"
    )

    # Download the file
    image_bytes = gcs_client.download_bytes(gcs_path=uri)
