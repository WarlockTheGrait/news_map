from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings

from spiders import tv1_spider


def run():
    settings = get_project_settings()
    settings.set("FEED_EXPORT_ENCODING", 'utf-8')
    settings.set("CONCURRENT_REQUESTS", 32)
    settings.set("ITEM_PIPELINES", {'utils.pipelines.AddTablePipeline': 100})

    configure_logging()
    process = CrawlerProcess(settings)
    process.crawl(tv1_spider.Tv1)
    process.start()


if __name__ == '__main__':
    run()
