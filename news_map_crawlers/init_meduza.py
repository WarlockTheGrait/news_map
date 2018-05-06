from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
from twisted.internet import reactor

from spiders import meduza_scrape


def run():
    settings = get_project_settings()
    settings.set("FEED_EXPORT_ENCODING", 'utf-8')
    settings.set("CONCURRENT_REQUESTS", 32)
    settings.set("ITEM_PIPELINES", {'pipelines.pipelines_meduza.AddTablePipeline': 100})


    configure_logging()
    runner = CrawlerRunner(settings)
    runner.crawl(meduza_scrape.Meduza)

    d = runner.join()
    d.addBoth(lambda _: reactor.stop())

    reactor.run()  # the script will block here until all crawling jobs are finished


if __name__ == '__main__':
    run()
