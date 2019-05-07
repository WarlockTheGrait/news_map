import re
from dateutil import parser


months = {'января': 'january', 'февраля': 'february', 'марта': 'march',
          'апреля': 'april', 'мая': 'may', 'июня': 'june',
          'июля': 'july', 'августа': 'august', 'сентября': 'september',
          'октября': 'october', 'ноября': 'november', 'декабря': 'december'}


def str2date(str_date):
    try:
        date = parser.parse(str_date)
    except:
        m = re.findall(r'[А-Яа-я]+', str_date)
        if m:
            date = parser.parse(re.sub(m[0], months[m[0]], str_date))
        else:
            date = None

    return date