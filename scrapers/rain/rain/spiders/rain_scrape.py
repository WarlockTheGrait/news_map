import scrapy
from rain.items import RainItem
from datetime import datetime
import re
import time


class Rain(scrapy.Spider):
	name = "rain_scraper"

	start_urls = []
	N = 4

	for i in range(1, N + 1):
		start_urls.append("https://tvrain.ru/archive/?tab=News&page="+str(i)+"")
	
	def parse(self, response):
		for href in response.xpath("//div[contains(@class, 'chrono_list__item__info')]/a[contains(@class, 'chrono_list__item__info__name chrono_list__item__info__name--nocursor')]//@href"):
			# add the scheme, eg http://
			url  = "https://tvrain.ru" + href.extract()
			time.sleep(3)
			yield scrapy.Request(url, callback=self.parse_dir_contents)	
					
	def parse_dir_contents(self, response):
		item = RainItem()

		item['Title'] = " ".join(response.xpath("//div[contains(@class, 'document-head__title')]/descendant::text()").extract()).strip()

		item['Date']= " ".join(response.xpath("//span[contains(@class, 'document-head__date')]/descendant::text()").extract()).strip()

		item['articleLead'] = " ".join(response.xpath("//div[contains(@class, 'document-lead')]/descendant::text()").extract()).strip()

		item['articleText']  = " ".join(response.xpath("//div[contains(@class, 'article-full__text')]").xpath(".//p/descendant::text()").extract()).strip()

		# Url (The link to the page)
		item['url'] = response.xpath("//meta[@property='og:url']/@content").extract()

		yield item

