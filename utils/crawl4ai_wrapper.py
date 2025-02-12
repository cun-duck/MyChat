from crawl4ai import AsyncWebCrawler
import asyncio

def crawl_url(url):
    """
    Crawls the given URL using the crawl4ai repository.
    Returns a dictionary containing the extracted text.
    """
    async def run_crawler():
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return {"text": result.markdown}

    return asyncio.run(run_crawler())
