from scrapy import Spider, Request
from scrapy.exceptions import CloseSpider
from dateutil import rrule
from datetime import datetime, date

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import func

from utils.items import ArticleItem
from utils.database_classes import Base, RawRain
from utils.convert_date import str2date


class Rain(Spider):
    name = "rain_spider"
    handle_httpstatus_list = [404]
    start_urls = []
    Base = Base
    Table = RawRain

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        engine = create_engine(r'sqlite:///News_map_database.db')

        self.db = scoped_session(sessionmaker(autocommit=False,
                                              autoflush=False,
                                              bind=engine))
        if engine.has_table(self.Table.__tablename__):
            self.Table.__table__.drop(engine)

        self.Base.metadata.create_all(engine)

        # last_date = self.db.query(func.max(self.Table.Date)).first()[0]
        last_date = date(2014, 1, 1)
        now = datetime.now()

        for dt in rrule.rrule(rrule.MONTHLY, dtstart=last_date, until=now):
            url = "https://tvrain.ru/archive/?search_year={}&search_month={}&search_day=0&query=&type=&tab=News&page=1".format(
                dt.year, dt.month)
            self.start_urls.append(url)


    def parse(self, response):
        N = [int(s.strip()) for s in response.xpath(
            "//a[contains(@class, 'pagination__item pagination__item--link')]//text()").extract()][-1]
        pattern = response.request.url[:-1]
        for i in range(1, N + 1):
            url = pattern + str(i)
            yield Request(url, callback=self.parse_pages)

    # парсим страницы
    def parse_pages(self, response):
        for href in response.xpath(
                "//div[contains(@class, 'chrono_list__item__info')]/a[contains(@class, 'chrono_list__item__info__name chrono_list__item__info__name--nocursor')]//@href"):
            # add the scheme, eg http://
            url = response.urljoin(href.extract())
            yield Request(url, callback=self.parse_articles)

    # парсим статьи
    def parse_articles(self, response):
        # if response.status == 404:
            # raise CloseSpider('404 error')
        if response.xpath("//div[@class = 'document-head__f1__l']/a/@href").extract_first() == '/news/':
            item = ArticleItem()

            item['Title'] = " ".join(
                response.xpath("//div[contains(@class, 'document-head__title')]//text()").extract()).strip()

            str_date = " ".join(
                response.xpath("//span[contains(@class, 'document-head__date')]//text()").extract()).strip()

            item['Date'] = str2date(str_date)

            lead = " ".join(
                response.xpath("//div[contains(@class, 'document-lead')]//text()").extract()).strip()

            item['Text'] = lead + " " + " ".join(
                response.xpath("//div[contains(@class, 'article-full__text')]").xpath("p//text()").extract()).strip()

            # Url (The link to the page)
            item['url'] = response.xpath("//meta[@property='og:url']/@content").extract_first()

            yield item
