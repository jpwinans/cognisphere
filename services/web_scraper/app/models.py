from typing import List

from pydantic import BaseModel


class ScrapeUrlsRequest(BaseModel):
    urls: List[str]
    category: str = "webpage"
    traverse: bool = False


class ScrapeUrlsRequest(BaseModel):
    urls: List[str]
    category: str = "webpage"
    traverse: bool = False
