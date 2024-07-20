Date and time
=============

The first step in setting up the environment is to specify the date and time of
the flight that will be simulated.

Basic usage
-----------

RocketPy requires detailed date and time information, including the year, month,
day, and hour, to accurately forecast the weather.
Additionally, a timezone can be specified to localize the time.

.. jupyter-execute::

    from rocketpy import Environment
    
    env = Environment(
        date=(2022, 2, 16, 18), # year, month, day, hour 
        timezone="America/New_York"
    )

.. tip::
    
    To view a list of available timezones, use the following code:

    .. code-block:: python

        import pytz

        print(pytz.all_timezones)

It is also possible to omit the timezone, in which case RocketPy will assume the
time is given in standard UTC time.

.. seealso::

    For more information on timezones, see `pytz <https://pypi.org/project/pytz/>`_.


Using ``datetime`` objects
--------------------------

The ``date`` argument also supports ``datetime`` objects, which are python
built-in objects that represent a date and time.


.. jupyter-execute::

    from datetime import datetime
    from rocketpy import Environment

    date = datetime(2022, 2, 16, 18)
    env = Environment(date=date)

Setting tomorrow's date
-----------------------

In the examples we will cover next, it is quite common to set the launch date to
tomorrow. This can be done by using the ``datetime`` module to calculate the
date of tomorrow and then passing it to the Environment class.

.. jupyter-execute::

    from datetime import date, timedelta

    tomorrow = date.today() + timedelta(days=1)

    date_info = (tomorrow.year, tomorrow.month, tomorrow.day, 9)  # Hour given in UTC time

    print("Tomorrow's date:", date_info)

.. note::

    The ``datetime`` module in Python contains both the ``date`` and ``timedelta`` \
    submodules, as well as the ``datetime`` class. The ``datetime`` class is \
    used to create datetime objects. It is common to confuse the ``datetime`` \
    module (which includes various classes and methods for manipulating dates \
    and times) with the ``datetime`` class itself (which is used to create \
    specific datetime objects). \


Alternatively, you can use the ``datetime`` module to calculate the date of tomorrow and then pass it to the Environment class.

.. jupyter-execute::

    from datetime import datetime, timedelta
    from rocketpy import Environment

    tomorrow = datetime.now() + timedelta(days=1)

    env = Environment(date=tomorrow)

