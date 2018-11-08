import vk

from scrapy.http.request import Request
from scrapy import Spider
from items import ArticleItem


class Meduza(Spider):
    name = "meduza_spider"

    def start_requests(self):
        session = vk.Session()
        vk_api = vk.API(session, access_token='86fbb29186fbb29186fbb291f786c179ee886fb86fbb291dccff2b3f1168419e0c7e829',
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
            wall = vk_api.wall.get(domain='meduzaproject', count=100, offset=offset)

    def parse(self, response):
        item = ArticleItem()
        item['Title'] = response.xpath("//span[@class = 'NewsMaterialHeader-first']//text()").extract_first()
        item['Date'] = response.xpath("//div[@class = 'MaterialMeta MaterialMeta--time']//text()").extract_first()
        item['Text'] = " ".join(response.xpath("//div[@class = 'Body']").xpath(".//p//text()").extract())
        item['url'] = response.xpath("//meta[@property='og:url']/@content").extract_first()
        yield item
