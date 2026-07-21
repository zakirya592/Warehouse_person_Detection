import logging
import os
import threading
from datetime import datetime

import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class CloudinaryUploader:
    """Upload violation screenshots to Cloudinary."""

    def __init__(
        self,
        cloud_name=None,
        api_key=None,
        api_secret=None,
        folder="safety-violations",
    ):
        self.cloud_name = cloud_name or os.environ.get("CLOUDINARY_CLOUD_NAME")
        self.api_key = api_key or os.environ.get("CLOUDINARY_API_KEY")
        self.api_secret = api_secret or os.environ.get("CLOUDINARY_API_SECRET")
        self.folder = folder or os.environ.get("CLOUDINARY_FOLDER", "safety-violations")
        self.enabled = bool(self.cloud_name and self.api_key and self.api_secret)

        if self.enabled:
            cloudinary.config(
                cloud_name=self.cloud_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
                secure=True,
            )
            logger.info("Cloudinary uploader enabled (folder: %s)", self.folder)
        else:
            logger.warning(
                "Cloudinary uploader disabled. Set CLOUDINARY_CLOUD_NAME, "
                "CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET to enable uploads."
            )

    def upload(self, image_path, tags=None, context=None):
        """
        Upload an image file to Cloudinary.

        Returns:
            dict with url/public_id on success, or None on failure/disabled.
        """
        if not self.enabled:
            return None

        if not image_path or not os.path.isfile(image_path):
            logger.error("Cloudinary upload skipped: file not found (%s)", image_path)
            return None

        filename = os.path.basename(image_path)
        public_id = os.path.splitext(filename)[0]

        try:
            result = cloudinary.uploader.upload(
                image_path,
                folder=self.folder,
                public_id=public_id,
                resource_type="image",
                tags=tags or ["safety-violation", "ppe-detection"],
                context=context or {},
                overwrite=True,
            )
            upload_result = {
                "url": result.get("secure_url") or result.get("url"),
                "public_id": result.get("public_id"),
                "bytes": result.get("bytes"),
            }
            logger.info("Cloudinary upload complete: %s", upload_result["url"])
            return upload_result
        except Exception as exc:
            logger.error("Cloudinary upload failed for %s: %s", image_path, exc)
            return None

    def upload_async(self, image_path, on_complete=None, tags=None, context=None):
        """Upload in a background thread so detection is not blocked."""

        def _worker():
            result = self.upload(image_path, tags=tags, context=context)
            if on_complete:
                on_complete(result)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return thread

    @staticmethod
    def build_context(camera_name=None, violations=None):
        return {
            "camera": camera_name or "unknown",
            "violations": ",".join(violations or []),
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        }
