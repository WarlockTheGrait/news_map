import scrapy
from items import ArticleItem
from datetime import datetime
import re
import time
import vk_api
from scrapy.http.request import Request


class Meduza(scrapy.Spider):
    name = "meduza_scraper"

    def start_requests(self):
        login, password = '*******', '*********'
        vk_session = vk_api.VkApi(login, password)

        try:
            vk_session.auth()
        except vk_api.AuthError as error_msg:
            print(error_msg)

        tools = vk_api.VkTools(vk_session)

        wall_iter = tools.get_all_iter('wall.get', 100, {'domain': 'meduzaproject'})

        k = 0

        for post in wall_iter:
            try:
                url = post['attachments'][0]['link']['url']
            except:
                pass
            else:
                if url[:23] == 'https://meduza.io/news/':
                    k += 1
                    yield Request(url, self.parse)
            if k > 200: # при любом к скачивается на 5 статей меньше, думаю, это связанно с асинхронностью
                break

    def parse(self, response):
        item = ArticleItem()
        item['Title'] = response.xpath("//span[@class = 'NewsMaterialHeader-first']/descendant::text()").extract()[0]
        item['Date'] = response.xpath("//div[@class = 'MaterialMeta MaterialMeta--time']/descendant::text()").extract()[
            0]
        item['Text'] = " ".join(response.xpath("//div[@class = 'Body']").xpath(".//p/descendant::text()").extract())
        item['url'] = response.xpath("//meta[@property='og:url']/@content").extract()[0]
        yield item
