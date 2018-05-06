# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html


from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
import dateutil.parser

engine = create_engine(r'sqlite:///News_map_database.db')



db = scoped_session(sessionmaker(autocommit=False,
                                 autoflush=False,
                                 bind=engine))
Base = declarative_base()

class Data(Base):
    __tablename__ = 'Lenta_table'

    id = Column(Integer, primary_key=True)
    Title = Column(String)
    Text = Column(String)
    Date = Column(DateTime)
    url = Column(String)

    def __init__(self, id=None, Title=None, Text = None, Date = None, url=None):
        self.id = id
        self.Title = Title
        self.url = url
        self.Text = Text
        self.Date = dateutil.parser.parse(Date)

    def __repr__(self):
        return "<Lenta_table: id='%d', Title='%s', Text='%s', Date = '%s', url = '%s'>" % \
               (self.id, self.Title, self.Text, str(self.Date), self.url)

if engine.has_table('Lenta_table'):  #удалить в случае дозаписи
    Data.__table__.drop(engine)


Base.metadata.create_all(engine)


class AddTablePipeline(object):

    def process_item(self, item, spider):
        if (item['Title'] and item['Date'] and item['Text']):
        # create a new SQL Alchemy object and add to the db session
            record = Data(Title=item['Title'],
                             Text=item['Text'],
                             Date= item['Date'],
                             url= item['url'])
            db.add(record)
            db.commit()
            return item