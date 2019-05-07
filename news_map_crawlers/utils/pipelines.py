from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from scrapy.exceptions import DropItem


class AddTablePipeline(object):
    """
    def open_spider(self, spider):
        Base = spider.Base
        self.Table = spider.Table
        engine = create_engine(r'sqlite:///News_map_database.db')

        self.db = scoped_session(sessionmaker(autocommit=False,
                                              autoflush=False,
                                              bind=engine))

        # if engine.has_table(self.Table.__tablename__):
            # self.Table.__table__.drop(engine)

        Base.metadata.create_all(engine)
        """

    def process_item(self, item, spider):
        # if not spider.db.query(spider.Table).filter(spider.Table.Title == item['url']).first() is None:
            # raise DropItem("Duplicated article: %s" % item)
        if item['Title'] and item['Date'] and item['Text']:
            # create a new SQL Alchemy object and add to the db session
            record = spider.Table(Title=item['Title'],
                                  Text=item['Text'],
                                  Date=item['Date'],
                                  url=item['url'])
            spider.db.add(record)
            spider.db.commit()
            return item
        else:
            print("Some fields missed")
            raise DropItem("Some fields missed: %s" % item)
