# Standard Library Modules
import calendar
import datetime


def get_unixtime():
    dt = datetime.datetime.utcnow()
    return calendar.timegm(dt.utctimetuple()) + (dt.microsecond * .000001)


def get_date_string():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def get_date_filename():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def get_time_string():
    return datetime.datetime.utcnow().strftime("%H:%M:%S")


def get_time_filename():
    return datetime.datetime.utcnow().strftime("%H-%M-%S")


def get_datetime():
    return datetime.datetime.utcnow()


def get_datetime_string(mantissa=False):
    if mantissa:
        return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S.%f")
    else:
        return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")


def get_datetime_filename(mantissa=False):
    if mantissa:
        return datetime.datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S_%f")
    else:
        return datetime.datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")


def get_datetime_filename_underscores(mantissa=False):
    if mantissa:
        return datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")
    else:
        return datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")


def datetime_to_unixtime(datetime_filename):
    year, month, day, hour, minute, second, microsecond = [int(x) for x in datetime_filename.split('_')]
    unix_time = datetime.datetime(year, month, day, hour, minute, second, microsecond).timestamp()
    return unix_time


def unixtime_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%Y_%m_%d_%H_%M_%S_%f")


def unixtime_to_datetime_filename(unix_time, mantissa=True):
    if mantissa:
        return datetime.datetime.fromtimestamp(unix_time).strftime("%Y_%m_%d_%H_%M_%S_%f")
    else:
        return datetime.datetime.fromtimestamp(unix_time).strftime("%Y_%m_%d_%H_%M_%S")

