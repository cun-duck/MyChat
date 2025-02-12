from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import asyncio

def crawl_url(url):
    """
    Crawls the given URL using the crawl4ai repository.
    Returns a dictionary containing the extracted text.
    """
    async def run_crawler():
        browser_config = BrowserConfig(headless=True, verbose=True)
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed", min_word_threshold=0)
            )
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            return {"text": result.markdown}

    return asyncio.run(run_crawler())
