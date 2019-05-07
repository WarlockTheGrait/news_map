import vk
import re

from time import sleep

from scrapy.http.request import Request
from scrapy import Spider
from scrapy.exceptions import CloseSpider

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from utils.items import ArticleItem
from utils.database_classes import Base, RawTv1
from utils.convert_date import str2date


class Tv1(Spider):
    name = "tv1_spider"
    Base = Base
    Table = RawTv1

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
        vk_api = vk.API(session, access_token='4ed4c464c0fadfb8ed26966abe533c03946d4ee448b8d028f242f324b71384e8ee939e5323fece345b388',
                        v=5.26)

        wall = vk_api.wall.get(domain='1tvnews', count=100)
        #wall = self.get_wall(vk_api, 0)
        count = wall['count']
        for offset in range(100, count + 100, 100):
            for post in wall['items']:
                try:
                    url = re.search('(?P<url>https?://[^\s]+)', post['text']).group("url")
                except:
                    pass
                else:
                    if url.startswith('https://www.1tv.ru/n'):
                        yield Request(url, self.parse)
            sleep(1)
            wall = vk_api.wall.get(domain='1tvnews', count=100, offset=offset)

    def get_wall(self, vk_api, offset):
        try:
            wall = vk_api.wall.get(domain='1tvnews', count=100, offset=offset)
        except:
            sleep(1)
            return self.get_wall(vk_api, offset)
        return wall

    def parse(self, response):
        if response.status == 404:
            raise CloseSpider('404 error')
        item = ArticleItem()
        item['Title'] = response.xpath("//h1[@class = 'title']//text()").extract_first()
        str_date = response.xpath("//div[@class ='date']//text()").extract_first()
        item['Date'] = str2date(str_date)
        item['Text'] = " ".join(response.xpath("//div[@data-block='w_text_block']").xpath(".//p//text()").extract())
        item['url'] = response.xpath("//meta[@property='og:url']/@content").extract_first()
        yield item
