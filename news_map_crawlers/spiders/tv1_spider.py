import re
import vk

from scrapy import Spider
from scrapy.http.request import Request

from items import ArticleItem


class Tv1(Spider):
    name = "tv1_spider"

    def start_requests(self):
        session = vk.Session()
        vk_api = vk.API(session, access_token='86fbb29186fbb29186fbb291f786c179ee886fb86fbb291dccff2b3f1168419e0c7e829',
                        v=5.26)

        wall = vk_api.wall.get(domain='1tvnews', count=100)
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
                wall = vk_api.wall.get(domain='1tvnews', count=100, offset=offset)

    def parse(self, response):
        item = ArticleItem()
        item['Title'] = response.xpath("//h1[@class = 'title']//text()").extract_first()
        item['Date'] = response.xpath("//div[@class ='date']//text()").extract_first()
        item['Text'] = " ".join(
            response.xpath("//div[@class ='editor text-block active']").xpath(".//p//text()").extract())
        item['url'] = response.xpath("//meta[@property='og:url']/@content").extract_first()
        yield item
