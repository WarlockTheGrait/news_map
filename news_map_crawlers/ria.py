from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings

from spiders import ria_spider


def run():
    settings = get_project_settings()
    settings.set("FEED_EXPORT_ENCODING", 'utf-8')
    settings.set("CONCURRENT_REQUESTS ", 1)
    settings.set("ITEM_PIPELINES", {'utils.ner.NerPipeline': 100})
    settings.set("DOWNLOADER_MIDDLEWARES", {
        'scrapy.downloadermiddleware.useragent.UserAgentMiddleware': None,
        'utils.middlewares.RandomUserAgentMiddleware': 400,
    })
    settings.set("DOWNLOAD_DELAY", 0.25)

    configure_logging()
    process = CrawlerProcess(settings)
    process.crawl(ria_spider.Ria)

    process.start()


if __name__ == '__main__':
    run()
