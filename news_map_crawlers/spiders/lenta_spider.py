from dateutil import parser
from scrapy import Spider, Request
from scrapy.exceptions import CloseSpider
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import func

from utils.items import ArticleItem
from utils.database_classes import Base, RawLenta


class Lenta(Spider):
    name = 'lenta_spider'
    handle_httpstatus_list = [404]
    Table = RawLenta
    Base = Base

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        engine = create_engine(r'sqlite:///News_map_database.db')

        self.db = scoped_session(sessionmaker(autocommit=False,
                                              autoflush=False,
                                              bind=engine))

        # if engine.has_table(self.Table.__tablename__):
            # self.Table.__table__.drop(engine)

        self.Base.metadata.create_all(engine)

        self.date = self.db.query(func.max(self.Table.Date)).first()[0]

        url = f"https://lenta.ru/news/2013/01/01"

        self.start_urls = [url]

    def parse(self, response):
        for href in response.xpath("//div[contains(@class, 'titles')]//a//@href"):
            url = response.urljoin(href.extract())
            # if self.db.query(self.Table).filter(self.Table.url == url).first() is None:
            yield Request(url, callback=self.parse_articles)

        list_pages = response.xpath("//div[contains(@class, 'b-archive__header-date')]//a//@href").extract()
        if len(list_pages) == 3:  # на последней странице нет кнопки next, поэтому кнопок меньше
            next_page = response.urljoin(list_pages[-1])
            if next_page.startswith('https://lenta.ru/news'):
                yield Request(next_page, callback=self.parse_next)

    def parse_next(self, response):
        for href in response.xpath("//div[contains(@class, 'titles')]//a//@href"):
            url = response.urljoin(href.extract())
            yield Request(url, callback=self.parse_articles)

        list_pages = response.xpath("//div[contains(@class, 'b-archive__header-date')]//a//@href").extract()
        if len(list_pages) == 3:  # на последней странице нет кнопки next, поэтому кнопок меньше
            next_page = response.urljoin(list_pages[-1])
            if next_page.startswith('https://lenta.ru/news'):
                yield Request(next_page, callback=self.parse_next)

    def parse_articles(self, response):
        if response.status == 404:
            raise CloseSpider('404 error')
        item = ArticleItem()

        item['Title'] = response.xpath('//h1//text()').extract_first()

        item['Date'] = parser.parse(
            response.xpath("//div[@class= 'b-topic__info']//@datetime").extract_first())

        item['Text'] = " ".join(response.xpath("//div[@itemprop='articleBody']/p/text()").extract()).strip()

        item['url'] = response.xpath("//meta[@property='og:url']/@content").extract_first()

        yield item



