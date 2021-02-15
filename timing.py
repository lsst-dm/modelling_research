# This file is part of modelling_research.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from timeit import default_timer as timer


def time_print(time=None, time_new=None, format_time='.1f', prefix='', postfix='', unit='s'):
    """ Print a message along with a time interval since a previous time.

    Parameters
    ----------
    time : time-like
        A previous time. Ignored if None.
    time_new : time-like
        A subsequent time of a compatible type as `time`. Default timeit.default_timer().
    format_time : `str`
        A string format for the time.
    prefix : `str`
        A string to prepend to the time; default empty.
    postfix : `str`
        A string to append to the time; default empty.
    unit : `str`
        The time unit to postfix to the timer output; default "s".

    Returns
    -------
    time_new : time-like
        The subsequent time.
    """
    if time_new is None:
        time_new = timer()
    timing = '' if time is None else (
        f'{time_new-time:{format_time}}{unit}' if (format_time is not None) else f'{time_new-time}{unit}')
    print(f'{prefix}{timing}{postfix}')
    return time_new
