from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings

from spiders import rain_spider


def run():
    settings = get_project_settings()
    settings.set("FEED_EXPORT_ENCODING", 'utf-8')
    settings.set("CONCURRENT_REQUESTS", 32)
    settings.set("ITEM_PIPELINES", {'pipelines.pipelines_rain.AddTablePipeline': 100})
    settings.set("DOWNLOADER_MIDDLEWARES", {
        'scrapy.downloadermiddleware.useragent.UserAgentMiddleware': None,
        'scrapy_fake_useragent.middlewares.RandomUserAgentMiddleware': 400,
    })

    configure_logging()
    process = CrawlerProcess(settings)
    process.crawl(rain_spider.Rain)
    process.start()


if __name__ == '__main__':
    run()
