from dateutil import parser

from scrapy import Spider, Request
from scrapy.exceptions import CloseSpider
from dateutil import rrule
from datetime import datetime, date

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import func

from utils.items import ArticleItem
from utils.database_classes import RiaBase, RiaNer
from utils.convert_date import str2date


class Ria(Spider):
    name = "ria_spider"
    handle_httpstatus_list = [404]
    start_urls = []
    Base = RiaBase
    Table = RiaNer

    time_list = ['020000', '040000', '060000', '080000', '090000', '093000', '100000', '103000',
                 '190000', '193000', '200000', '210000', '220000', '230000', '240000']
    for i in range(11, 19):
        time_list.append(f'{i}0000')
        time_list.append(f'{i}2000')
        time_list.append(f'{i}4000')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        engine = create_engine(r'sqlite:///Ria_ner_database.db')

        self.db = scoped_session(sessionmaker(autocommit=False,
                                              autoflush=False,
                                              bind=engine))
        if engine.has_table(self.Table.__tablename__):
            self.Table.__table__.drop(engine)

        self.Base.metadata.create_all(engine)

        # last_date = self.db.query(func.max(self.Table.Date)).first()[0]
        last_date = date(2017, 1, 1)
        print(last_date)
        now = datetime.now()

        for dt in rrule.rrule(rrule.DAILY, dtstart=last_date, until=now):
            if dt.month > 9:
                month = dt.month
            else:
                month = '0' + str(dt.month)
            if dt.day > 9:
                day = dt.day
            else:
                day = '0' + str(dt.day)
            url = f'https://ria.ru/{dt.year}{month}{day}/'
            self.start_urls.append(url)

    def parse(self, response):
        raw_url = response.urljoin(response.xpath('//div[@class="list-more"]/@data-url').extract_first()).split('T')[0]
        for time in self.time_list:
            yield Request(f'{raw_url}T{time}', callback=self.parse_service)

    def parse_service(self, response):
        lst = response.xpath('//div[@class="list-item"]/a/@href').extract()
        if lst:
            for url in lst:
                if url.startswith('https://ria.ru/'):
                    yield Request(url, callback=self.parse_articles)

    def parse_articles(self, response):
        # if response.status == 404:
            # raise CloseSpider('404 error')
        item = ArticleItem()

        item['Title'] = ''

        item['Date'] = parser.parse(response.xpath("//div[@class='endless']/div/@data-published").extract_first())

        item['Text'] = ' '.join(response.xpath("//div[@class='article__text']//text()").extract())

        # Url (The link to the page)
        item['url'] = ''

        yield item
