import vk
import dateutil.parser
import re

from time import sleep

from scrapy.http.request import Request
from scrapy import Spider
from scrapy.exceptions import CloseSpider

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from utils.items import ArticleItem
from utils.database_classes import Base, RawMeduza
from utils.convert_date import str2date


class Meduza(Spider):
    name = "meduza_spider"
    Base = Base
    Table = RawMeduza

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        engine = create_engine(r'sqlite:///News_map_database.db')

        self.db = scoped_session(sessionmaker(autocommit=False,
                                              autoflush=False,
                                              bind=engine))
        if engine.has_table(self.Table.__tablename__):
            self.Table.__table__.drop(engine)

        self.Base.metadata.create_all(engine)

    def start_requests(self):
        session = vk.Session()
        vk_api = vk.API(session,
                        access_token='4ed4c464c0fadfb8ed26966abe533c03946d4ee448b8d028f242f324b71384e8ee939e5323fece345b388',
                        v=5.26)
        wall = vk_api.wall.get(domain='meduzaproject', count=100)
        count = wall['count']
        for offset in range(100, count+100, 100):
            for post in wall['items']:
                try:
                    url = post['attachments'][0]['link']['url']
                except:
                    pass
                else:
                    if url.startswith('https://meduza.io/news/'):
                        yield Request(url, self.parse)
            sleep(1)
            wall = vk_api.wall.get(domain='meduzaproject', count=100, offset=offset)

    def parse(self, response):
        #  if response.status == 404:
            # raise CloseSpider('404 error')
        item = ArticleItem()
        item['Title'] = response.xpath("//h1[@class = 'SimpleTitle-root']//text()").extract_first()

        str_date = response.xpath("//time[@class = 'Timestamp-root']//text()").extract_first()
        item['Date'] = str2date(str_date)

        item['Text'] = " ".join(response.xpath("//div[@class = 'GeneralMaterial-article']").xpath(".//p//text()").extract())
        item['url'] = response.xpath("//meta[@property='og:url']/@content").extract_first()
        yield item
