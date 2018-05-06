# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
import dateutil.parser
from datetime import datetime, date
import re

engine = create_engine(r'sqlite:///News_map_database.db')

db = scoped_session(sessionmaker(autocommit=False,
                                 autoflush=False,
                                 bind=engine))

Base = declarative_base()
months = {'января': 'january', 'февраля': 'february', 'марта': 'march', 'апреля': 'april', 'мая': 'may', 'июня': 'june',
          'июля': 'july', 'августа': 'august', 'сентября': 'september', 'октября': 'october', 'ноября': 'november',
          'декабря': 'december'}


class Data(Base):
    __tablename__ = 'Rain_table'

    id = Column(Integer, primary_key=True)
    Title = Column(String)
    Text = Column(String)
    Date = Column(DateTime)
    url = Column(String)

    def __init__(self, id=None, Title=None, Text=None, Date=None, url=None):
        self.id = id
        self.Title = Title
        self.url = url
        self.Text = Text
        try:
            self.Date = dateutil.parser.parse(Date)
        except:
            m = re.findall(r'[А-Яа-я]+', Date)
            if m:
                temp = dateutil.parser.parse(re.sub(m[0], months[m[0]], Date))
                self.Date = temp

            else:
                self.Date = NULL  # я проверяла январь, апрель и май, ни разу не было NULL

    def __repr__(self):
        return "<%s: id='%d', Title='%s', Text='%s', Date = '%s', url = '%s'>" % \
               (self.__tablename__, self.id, self.Title, self.Text, str(self.Date), self.url)


# удалить в случае дозаписи
if engine.has_table('Rain_table'):
    Data.__table__.drop(engine)

Base.metadata.create_all(engine)


class AddTablePipeline(object):

    def process_item(self, item, spider):
        if (item['Title'] and item['Date'] and item['Text']):
            # create a new SQL Alchemy object and add to the db session
            record = Data(Title=item['Title'],
                          Text=item['Text'],
                          Date=item['Date'],
                          url=item['url'])
            db.add(record)
            db.commit()
            return item
