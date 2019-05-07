# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy

class ArticleItem(scrapy.Item):
    Title = scrapy.Field()
    Date = scrapy.Field()
    Text = scrapy.Field()
    url = scrapy.Field()
