import argparse
import datetime
import time
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
from dateutil import parser

DATE_FORMAT = '%Y-%m-%d'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
SECONDS_IN_DAY = 3600 * 24

def validate_datetime_format(s: Optional[str], fmt: str) -> Optional[datetime.datetime]:
    """
    :param s: string, command-line argument
    :param fmt: expected format
    :return: parsed datetime in given format fmt if s is not empty
    """
    if not s:
        return None
    try:
        return datetime.datetime.strptime(s, fmt)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a valid date(time): '{s}'. Expected format '{fmt}'.")


def valid_date(s: str) -> Optional[datetime.datetime]:
    return validate_datetime_format(s, DATE_FORMAT)


def valid_date_str(s: str) -> Optional[str]:
    valid_date_value = validate_datetime_format(s, DATE_FORMAT)
    if valid_date_value:
        return valid_date_value.strftime(DATE_FORMAT)
    return None


def valid_datetime(s: str) -> Optional[datetime.datetime]:
    return validate_datetime_format(s, DATETIME_FORMAT)


def get_now_timestamp() -> int:
    return int(datetime.datetime.now().timestamp())


def date_range(
        begin_date_str: str,
        end_date_str: str,
        fmt: str = '%Y-%m-%d'
) -> List[str]:
    """
    :param begin_date_str: begin date string in format fmt, default format '%Y-%m-%d'
    :param end_date_str: end date string in same format
    :param fmt: format for datetime.datetime.strptime()
    :return:

    >>> date_range('2018-12-29', '2019-01-02')
    ['2018-12-29', '2018-12-30', '2018-12-31', '2019-01-01', '2019-01-02']
    """
    begin_date = datetime.datetime.strptime(begin_date_str, fmt)
    end_date = datetime.datetime.strptime(end_date_str, fmt)
    _range = []
    while begin_date <= end_date:
        _range.append(datetime.datetime.strftime(begin_date, fmt))
        begin_date += datetime.timedelta(days=1)
    return _range


def add_days_to_date(
        date_string: str,
        days: int,
        fmt: str = '%Y-%m-%d'
) -> str:
    return (datetime.datetime.strptime(date_string, fmt) + datetime.timedelta(days=days)).strftime(fmt)


def seconds_to_time_string(seconds: float) -> str:
    """
    :param seconds:  seconds since midnight
    :return: string in [D.]HH:MM:SS format

    >>> seconds_to_time_string(14400)
    '04:00:00'

    >>> seconds_to_time_string(86400)
    '1.00:00:00'

    """

    days_str = ''

    days = int(seconds) // SECONDS_IN_DAY
    if days:
        days_str = '{}.'.format(days)
    time_str = str(datetime.timedelta(seconds=int(seconds % SECONDS_IN_DAY)))
    if len(time_str) < len('00:00:00'):
        time_str = '0' + time_str
    return days_str + time_str


def time_string_to_seconds(time_string: str) -> float:
    """
    :param time_string: string in [D.]HH:MM:SS format
    :return: seconds, or None if format is incorrect

    >>> time_string_to_seconds('04:00:00')
    14400.0

    >>> time_string_to_seconds('1.00:00:00')
    86400.0
    """

    day_sep_pos = time_string.find('.')
    if day_sep_pos != -1:
        days = int(time_string[:day_sep_pos])
        time_string = time_string[day_sep_pos + 1:]
    else:
        days = 0

    time_sep_count = time_string.count(':')
    if time_sep_count == 2:
        t = datetime.datetime.strptime(time_string, '%H:%M:%S')
    elif time_sep_count == 1:
        t = datetime.datetime.strptime(time_string, '%H:%M')
    else:
        t = datetime.datetime.strptime(time_string, '%H')
    return datetime.timedelta(days=days, hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds()


def sec2str_absolute(seconds, date, time_zone):
    """
    Convert time defined as seconds since midnight of date in given time_zone, to ISO 8601 format
    in the same time_zone.
    :param seconds:  seconds since midnight (in time zone defined by time_zone argument)
    :param date: string in YYYY-MM-DD format
    :param time_zone: int, 0 for Greenwich, 3 for Moscow etc
    :return: string in ISO 8601 (YYYY-MM-DDTHH:MM:SS+ZZ:ZZ) format

    >>> sec2str_absolute(36000, '2020-10-01', 3)
    '2020-10-01T10:00:00+03:00'
    """
    midnight = parser.parse(date)
    dt = midnight + datetime.timedelta(seconds=seconds)
    return dt.strftime('%Y-%m-%dT%H:%M:%S{:+03d}:00'.format(time_zone))


def str2sec_absolute(time_string, date, time_zone):
    """
    Convert absolute time in ISO 8601 format format to seconds since midnight of date in provided time_zone.
    :param time_string: string in ISO 8601 (YYYY-MM-DDTHH:MM:SS+ZZ:ZZ) format
    :param date: string in YYYY-MM-DD format
    :param time_zone: int, 0 for Greenwich, 3 for Moscow etc
    """
    dt = parser.parse(time_string)
    local_midnight_dt = parser.parse(date + f'T00:00:00{time_zone:+03d}:00')
    return (dt - local_midnight_dt).total_seconds()


def time_absolute_to_relative(absolute_time, date, time_zone):
    """
    Convert time in ISO 8601 format to relative format [D.][HH:MM:SS], given date and time zone.
    If time zone in absolute_time is different from time_zone value, the time is shifted accordingly. For example:

    >>> time_absolute_to_relative('2020-10-16T07:30:00.000+03:00', '2020-10-16', 4)
    '08:30:00'

    Warning: if absolute_time is earlier than date, you'll get times with negative days like -1.05:00:00
    which the solver will not understand. If you try to use them in the solver, you'll get an error.
    :param absolute_time: 'YYYY-MM-DDTHH:MM:SS+ZZ:00'
    :param date: YYYY-MM-DD
    :param time_zone: integer, utc offset in hours (Moscow = 3)
    :return: [D.]HH:MM:SS
    """
    seconds = str2sec_absolute(absolute_time, date, time_zone)
    return seconds_to_time_string(seconds)


def time_window_absolute_to_relative(absolute_time_window, date, time_zone):
    """
    Convert absolute time window
    YYYY-MM-DDTHH:MM:SS+ZZ:00/YYYY-MM-DDTHH:MM:SS+ZZ:00 or YYYY-MM-DDTHH:MM:SS/YYYY-MM-DDTHH:MM:SS to relative
    HH:MM:SS-HH:MM:SS, given options.date and options.time_zone. See time_absolute_to_relative docsting for details.
    """
    bounds = absolute_time_window.split('/')
    return '-'.join(time_absolute_to_relative(bound, date, time_zone) for bound in bounds)


def add_seconds_to_time_string(time_string, seconds):
    """
    :param time_string: [D.]HH:MM:SS or YYYY-MM-DDTHH:MM:SS+ZZ:00 (ISO 8601)
    :param seconds: amount of seconds to add
    :return: string in the same format as time_string, and in same time zone if it's ISO 8601.
    """
    if 'T' in time_string:
        time_zone = int(parser.parse(time_string).utcoffset().total_seconds() / 3600)
        date = '2010-01-01'  # can be any date before our service was created
        return sec2str_absolute(str2sec_absolute(time_string, date, time_zone) + seconds, date, time_zone)
    else:
        return seconds_to_time_string(time_string_to_seconds(time_string) + seconds)


def time_window_str2dict(time_window_string):
    """

    :param time_window_string: a time window string in [D].HH[:MM[:SS]]-[D].HH[:MM[:SS]] format
    :return: dict of the form {'begin': window_begin_in_seconds, 'end': window_begin_in_seconds}

    >>> time_window_str2dict('04:00:00-1.00:00:00')
    {'begin': 14400.0, 'end': 86400.0}
    """

    time_window_string = time_window_string.replace(' ', '')
    begin, end = tuple(time_window_string.split('-'))
    return {'begin': time_string_to_seconds(begin), 'end': time_string_to_seconds(end)}


def time_window_string_to_dict(time_window_string, date, time_zone):
    """
    :param time_window_string: YYYY-MM-DDTHH:MM:SS+ZZ:00/YYYY-MM-DDTHH:MM:SS+ZZ:00
    :param date: task date in YYYY-MM-DD format
    :param time_zone: integer, task time zone (Moscow = 3)
    :return: {'begin': window_begin_in_seconds, 'end': window_begin_in_seconds}

    >>> time_window_string_to_dict("2020-10-16T11:00:00+03:00/2020-10-16T13:00:00+03:00", '2020-10-16', 3)
    {'begin': 39600.0, 'end': 46800.0}
    """
    begin, end = tuple(time_window_string.split('/'))
    kw = dict(date=date, time_zone=time_zone)
    return {'begin': str2sec_absolute(begin, **kw), 'end': str2sec_absolute(end, **kw)}


def time_window_dict2str(time_window_dict):
    """

    :param time_window_dict: a dict of the form {'begin': window_begin_in_seconds, 'end': window_begin_in_seconds}
    :return: time window string in [D].HH[:MM[:SS]]-[D].HH[:MM[:SS]] format

    >>> time_window_dict2str({'begin': 14400, 'end': 86400})
    '04:00:00-1.00:00:00'
    """

    return seconds_to_time_string(time_window_dict['begin']) + '-' + seconds_to_time_string(time_window_dict['end'])


def time_window_dict_to_string(time_window_dict, date, time_zone):
    kw = dict(date=date, time_zone=time_zone)
    return sec2str_absolute(time_window_dict['begin'], **kw) + '/' + sec2str_absolute(time_window_dict['end'], **kw)


def time_window_duration(time_window_string):
    """
    :param time_window_string: string in [D.]HH:MM:SS-[D.]HH:MM:SS format or in ISO format
    YYYY-MM-DDTHH:MM:SS+ZZ:00/YYYY-MM-DDTHH:MM:SS+ZZ:00 or YYYY-MM-DDTHH:MM:SS/YYYY-MM-DDTHH:MM:SS
    :return: time window duration in seconds
    """
    if '/' in time_window_string:
        time_window_string = time_window_absolute_to_relative(time_window_string, '1970-01-01', 0)
    time_window = time_window_str2dict(time_window_string)
    return time_window['end'] - time_window['begin']


def timestamp_to_seconds(ts, base_date_string):
    """

    :param ts: POSIX timestamp
    :param base_date_string: Date string in 'YYYY-MM-DD' format. Seconds are counted from 00:00:00 this date.
    :return: number of seconds since midnight of bas date to timestamp

    >>> timestamp_to_seconds(datetime.datetime(year=2018, month=10, day=9, hour=6), '2018-10-09')
    21600.0

    """
    reference_point = datetime.datetime.strptime(base_date_string, '%Y-%m-%d')
    return (ts - reference_point).total_seconds()


def timestamp_to_time_string(ts, base_date_string):
    return seconds_to_time_string(timestamp_to_seconds(ts, base_date_string))


def datetime_to_timestamp(dttm):
    return int(dttm.strftime('%s'))


def datetime_to_period_start(dt: datetime.datetime, period_name: str):
    """
    :param dt: datetime object
    :param period_name: one of 'day', 'week', 'month', 'quarter', 'hour'
    :return: beginning of the period containing dt
    """
    assert period_name in {'day', 'week', 'month', 'quarter', 'hour'}
    if period_name == 'quarter':
        month = dt.month - (dt.month - 1) % 3
        return datetime.datetime(year=dt.year, month=month, day=1)
    elif period_name == 'month':  # first day of the same month
        return datetime.datetime(year=dt.year, month=dt.month, day=1)
    elif period_name == 'week':  # monday of the same week
        return datetime.datetime(year=dt.year, month=dt.month, day=dt.day) - datetime.timedelta(days=dt.weekday())
    elif period_name == 'hour':  # monday of the same week
        return datetime.datetime(year=dt.year, month=dt.month, day=dt.day, hour=dt.hour)
    else:  # period_name == 'day'
        return datetime.datetime(year=dt.year, month=dt.month, day=dt.day)


def dt_to_timestamp(dt: datetime.datetime) -> float:
    return time.mktime(dt.timetuple()) + dt.microsecond / 1e6


def ts_to_str(ts: Optional[Union[int, float]]) -> Optional[str]:
    """
    :param ts: POSIX timestamp, int or float
    :return: 'YYYY-MM-DD HH:MM:SS[:ff]' string
    """
    if ts is None:
        return None
    else:
        return datetime.datetime.utcfromtimestamp(ts).isoformat(sep=' ', timespec='milliseconds')


def add_seconds_to_datetime_str(
        dt_str: str,
        seconds: int,
        sep: str = ' ',
        timespec: str = 'seconds'
) -> str:
    """
    Convenience function to change an ISO timestamp string by a given number of seconds
    :param dt_str: datetime string in ISO format with given separator sep and timespec.
    :param seconds: how many seconds to add
    :param sep: separator between date and time
    :param timespec: 'minutes', 'seconds', 'milliseconds', 'microseconds'
    :return: datetime string in same format
    """
    dt = datetime.datetime.fromisoformat(dt_str)
    offset_dt = dt + datetime.timedelta(seconds=seconds)
    return offset_dt.isoformat(sep=sep, timespec=timespec)


def split_datettime_interval(
        begin_dt_str: str,
        end_dt_str: str,
        freq: str = '2h'
) -> Iterator[Tuple[str, str]]:
    """
    Split a datetime interval into parts with given frequency.
    The first interval always starts at begin_dt_str, and the last one always ends at end_dt_str.
    :param begin_dt_str: 'YYYY-MM-DD HH:MM:SS', included
    :param end_dt_str: 'YYYY-MM-DD HH:MM:SS', not included
    :param freq: pandas time series offset alias, e.g. 'd' (1 day), '30min', '2h', default '2h'
        see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    :yield: prev_dt_str, next_dt_str
    >>> list(split_datettime_interval('2021-01-01 00:00:00', '2021-01-01 05:30:00', '2h'))
    [('2021-01-01 00:00:00', '2021-01-01 02:00:00'), ('2021-01-01 02:00:00', '2021-01-01 04:00:00'), ('2021-01-01 04:00:00', '2021-01-01 05:30:00')]
    """
    endpoints = list(pd.date_range(start=begin_dt_str, end=end_dt_str, freq=freq).astype(str))
    if endpoints[-1] < end_dt_str:
        endpoints.append(end_dt_str)
    for prev_endpoint, next_endpoint in zip(endpoints[:-1], endpoints[1:]):
        yield prev_endpoint, next_endpoint
