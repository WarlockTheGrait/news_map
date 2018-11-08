import scrapy
from items import ArticleItem


class Lenta(scrapy.Spider):
    name = 'lenta_spider'
    # стартовая страница, скачиваются все данные с заданного дня до этого момента
    start_urls = ["https://lenta.ru/news/2018/11/06"]

    def parse(self, response):
        for href in response.xpath("//div[contains(@class, 'titles')]//a//@href"):
            # add the scheme, eg http://
            url = response.urljoin(href.extract())
            yield scrapy.Request(url, callback=self.parse_articles)

        list_pages = response.xpath("//div[contains(@class, 'b-archive__header-date')]//a//@href").extract()
        if len(list_pages) == 3:  # на последней странице нет кнопки next, поэтому кнопок меньше
            next_page = response.urljoin(list_pages[-1])
            yield scrapy.Request(next_page, callback=self.parse)

    def parse_articles(self, response):
        item = ArticleItem()

        item['Title'] = response.xpath("//h1[@class= 'b-topic__title']//text()").extract_first()

        item['Date'] = response.xpath("//div[@class= 'b-topic__info']//@datetime").extract_first()

        item['Text'] = " ".join(
            response.xpath("//div[contains(@class, 'b-text clearfix js-topic__text')]").xpath(
                ".//p//text()").extract()).strip()

        item['url'] = response.xpath("//meta[@property='og:url']/@content").extract_first()

        yield item
