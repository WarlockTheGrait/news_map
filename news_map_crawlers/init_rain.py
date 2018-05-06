from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
from twisted.internet import reactor

from spiders import rain_scrape


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
    runner = CrawlerRunner(settings)
    runner.crawl(rain_scrape.Rain)

    d = runner.join()
    d.addBoth(lambda _: reactor.stop())

    reactor.run()  # the script will block here until all crawling jobs are finished


if __name__ == '__main__':
    run()
