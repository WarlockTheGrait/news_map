import scrapy
from items import ArticleItem
from datetime import datetime
import re
import time


class Lenta(scrapy.Spider):
    name = 'lenta_scraper'
    start_urls = []
    #стартовая страница, скачиваются все данные с заданного дня до этого момента
    start_urls.append("https://lenta.ru/news/2018/04/21")

    def parse(self, response):
        for href in response.xpath("//div[contains(@class, 'titles')]//a//@href"):
            # add the scheme, eg http://
            url = response.urljoin(href.extract())
            yield scrapy.Request(url, callback=self.parse_dir_contents)

        list_pages = response.xpath("//div[contains(@class, 'b-archive__header-date')]//a//@href").extract()
        if len(list_pages) == 3: # на последней странице нет кнопки next, поэтому кнопок меньше
            next_page = response.urljoin(list_pages[-1])
            yield scrapy.Request(next_page, callback=self.parse)

    def parse_dir_contents(self, response):
        item = ArticleItem()

        item['Title'] = response.xpath("//h1[@class= 'b-topic__title']//descendant::text()").extract_first()

        item['Date'] = response.xpath("//div[@class= 'b-topic__info']//@datetime").extract_first()

        item['Text'] = " ".join(response.xpath("//div[contains(@class, 'b-text clearfix js-topic__text')]").xpath(".//p/descendant::text()").extract()).strip()

        item['url'] = response.xpath("//meta[@property='og:url']/@content").extract()[0]

        yield item