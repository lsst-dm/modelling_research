from timeit import default_timer as timer


def time_print(time, time_new=None, format_time='.1f', prefix='', postfix=''):
    '''

    Parameters
    ----------
    time :
    format_time
    prefix
    postfix

    Returns
    -------

    '''
    if time_new is None:
        time_new = timer()
    timing = '' if time is None else (
        f'{time_new-time:{format_time}}s' if (format_time is not None) else f'{time_new-time}s')
    print(f'{prefix}{timing}{postfix}')
    return time_new