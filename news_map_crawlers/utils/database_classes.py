import dateutil.parser

from sqlalchemy import Column, String, Integer, DateTime, ARRAY
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declared_attr


Base = declarative_base()


class RawMixin(object):
    id = Column(Integer, primary_key=True)
    Title = Column(String)
    Text = Column(String)
    Date = Column(DateTime)
    url = Column(String)


class RawLenta(RawMixin, Base):
    __tablename__ = 'Lenta_table'


class RawMeduza(RawMixin, Base):
    __tablename__ = 'Meduza_table'


class RawRain(RawMixin, Base):
    __tablename__ = 'Rain_table'


class RawTv1(RawMixin, Base):
    __tablename__ = 'tv1_table'


class RawRia(RawMixin, Base):
    __tablename__ = 'Ria_table'


RiaBase = declarative_base()


class RiaNer(RiaBase):
    __tablename__ = 'Ria'
    id = Column(Integer, primary_key=True)
    Text = Column(String)
    Date = Column(DateTime)
    Person = Column(String)


NewBase = declarative_base()


class NewMixin:
    id = Column(Integer, primary_key=True)
    Text = Column(String)
    Person = Column(String)


class NewLenta(NewMixin, NewBase):
    __tablename__ = 'Lenta'


class NewMeduza(NewMixin, NewBase):
    __tablename__ = 'Meduza'


class NewRain(NewMixin, NewBase):
    __tablename__ = 'Rain'


class NewRia(NewMixin, NewBase):
    __tablename__ = 'Ria'