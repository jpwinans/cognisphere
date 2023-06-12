from typing import List

from pydantic import BaseModel


class ScrapeUrlsRequest(BaseModel):
    urls: List[str]
    categories: List[str] = ["webpage"]
    traverse: bool = False