import datetime

import numpy as np

RISE_FACTOR = 1.01
FALL_FACTOR = 0.5
FIRST_FLARE_FALL_FACTOR = 0.2


def apply_lyraff(clean_time, clean_irradiance, artifacts_removed=False):
    """
    Applies LYRAFF (LYRA Flare Finder) algorithm to a time series.

    Parameters
    ----------
    clean_time : ndarray/array-like of datetime objects
        measurement times with any relevant artifacts removed.

    clean_irradiance : ndarrays/array-like convertible to float64
        irradiance measurements with an relevant arifacts removed.

    artifacts_removed : `numpy.recarray` or False
        Details of any artifacts removed.  Must be of same structure as
        artifact_status["removed"] output by remove_lytaf_events().
        Default=False, i.e. no artifacts removed

    Returns
    -------
    lyra_flares : numpy recarray
        Contains the start, peak and end times and irradiance values for each
        event found.  The fields of the recarray are: 'start_time',
        'peak_time', 'end_time', 'start_irrad', 'peak_irrad', 'end_irrad'.

    Examples
    --------

    """
    # Ensure inputs are of correct type
    clean_irradiance = np.asanyarray(clean_irradiance, dtype="float64")
    clean_time = _check_datetime(clean_time)
    # Define recarray to store results
    lyra_flares = np.empty((0,), dtype=[("start_time", object),
                                        ("peak_time", object),
                                        ("end_time", object),
                                        ("start_irrad", float),
                                        ("peak_irrad", float),
                                        ("end_irrad", float),
                                        ("end_time_comment", 'a40')])
    # Iterate through time series to find where flare start criteria
    # are satisfied.
    end_series = len(clean_irradiance)-1
    i = 0
    while i < end_series-3:
        print "i = ", i, end_series
        # Check if current time satisfies flare reference start
        # criteria.
        if datetime.timedelta(seconds=150) <= \
          clean_time[i+3]-clean_time[i] <= datetime.timedelta(seconds=210) and \
          all(clean_irradiance[i+1:i+4]-clean_irradiance[i:i+3] > 0.) and \
          clean_irradiance[i+3] >= clean_irradiance[i]*RISE_FACTOR:
            # Find start time which is defined as earliest continuous
            # increase in irradiance before the point found by the
            # above criteria.
            k = i
            while clean_irradiance[k-1] < clean_irradiance[k] and k > 0:
                k = k-1
            start_index = k
            # If artifact is at start of flare, set start time to
            # directly afterwards.
            if artifacts_removed != False:
                artifact_check = np.logical_and(
                    artifacts_removed["end_time"] > clean_time[start_index],
                    artifacts_removed["end_time"] < clean_time[start_index+2])
                if artifact_check.any() == True:
                    artifact_at_start = artifacts_removed[artifact_check][-1]
                    start_index = np.where(clean_time > artifact_at_start["end_time"])[0][0]
            # Find time when flare signal descends by
            # FIRST_FLARE_FALL_FACTOR.
            j = i+4
            while clean_irradiance[j] > \
              np.max(clean_irradiance[start_index:j])-\
              (np.max(clean_irradiance[start_index:j])-\
              clean_irradiance[start_index])*FIRST_FLARE_FALL_FACTOR \
              and j < end_series: # and j < n-1:
                j = j+1
            # Once flare signal has descended by FIRST_FLARE_FALL_FACTOR,
            # start searching for when current flare ends or a new flare
            # starts.
            end_condition = False
            while end_condition == False and j < end_series:
                end_time_comment = ""
                # Check if flare has ended.
                if clean_irradiance[j] < \
                  np.max(clean_irradiance[i:j])-(np.max(clean_irradiance[i:j])-\
                  clean_irradiance[i])*FALL_FACTOR:
                    # Once flare reference end criteria have been met,
                    # find time of next increase in flare signal.
                    while clean_irradiance[j] > clean_irradiance[j+1]:
                        j = j+1
                    end_index = j
                    # If artifact is at end of flare, set end time
                    # to directly beforehand.
                    if artifacts_removed != False:
                        artifact_check = np.logical_and(
                          artifacts_removed["begin_time"] < clean_time[end_index],
                          artifacts_removed["begin_time"] > clean_time[end_index-2])
                        if artifact_check.any() == True:
                            artifact_at_end = artifacts_removed[artifact_check][0]
                            new_index = np.where(clean_time < artifact_at_end["begin_time"])
                            j = new_index[0][-1]
                            end_index = j
                    # Since flare end has been found, set end_condition to True
                    end_condition = True
                    end_time_comment = "Final end criterion found."
                # Check if another flare has started.
                elif j+3 < end_series-1 and datetime.timedelta(seconds=150) <= \
                  clean_time[j+3]-clean_time[j] <= datetime.timedelta(seconds=210) and \
                  all(clean_irradiance[j+1:j+4]-clean_irradiance[j:j+3] > 0.) and \
                  clean_irradiance[j+3] >= clean_irradiance[j]*RISE_FACTOR:
                    print j
                    # If a new flare is detected before current one
                    # ends, set end time of current flare to start time
                    # of next flare and iterate to move onto next flare.
                    kk = j
                    while clean_irradiance[kk-1] < clean_irradiance[kk]:
                        kk = kk-1
                    end_index = kk
                    end_condition = True
                    end_time_comment = "Another flare found during decay phase."
                # Else iterate and check next measurement.
                else:
                    j = j+1
            # Give warning if final flare end time caused by end of time series
            if j == end_series:
                end_index = j
                end_time_comment = "Time series ended before flare."
                raise warn(
                    "Final flare's end time caused by end of time series, not flare end criteria.")
            # Record flare.
            # Find peak index of flare
            peak_index = np.where(clean_irradiance[start_index:end_index] == \
                                  max(clean_irradiance[start_index:end_index]))
            peak_index = peak_index[0][0]+start_index
            # Record flare start, peak and end times
            lyra_flares = np.append(lyra_flares, np.empty(1, dtype=lyra_flares.dtype))
            lyra_flares[-1]["start_time"] = clean_time[start_index]
            lyra_flares[-1]["peak_time"] = clean_time[peak_index]
            lyra_flares[-1]["end_time"] = clean_time[end_index]
            lyra_flares[-1]["start_irrad"] = clean_irradiance[start_index]
            lyra_flares[-1]["peak_irrad"] = clean_irradiance[peak_index]
            lyra_flares[-1]["end_irrad"] = clean_irradiance[end_index]
            lyra_flares[-1]["end_time_comment"] = end_time_comment
            # Set principle iterator, i, to flare end index to begin
            # searching for next flare
            i = end_index
        else:
            # If this time does not satisfy start reference criteria,
            # try next measurement.
            i = i+1

    return lyra_flares


def _check_datetime(time):
    """
    Checks or tries to convert input array to array of datetime objects.

    Returns input time array with elements as datetime objects or raises an
    TypeError if time not of valid format.  Input format can be anything
    convertible to datetime by datetime() function or any time string valid as
    an input to sunpy.time.parse_time().

    """
    if (np.array([type(t) for t in time]) == datetime.datetime).all():
        new_time = np.asanyarray(time)
    elif type(time) == pandas.tseries.index.DatetimeIndex:
        new_time = time.to_pydatetime()
    else:
        # If elements of time are not datetime objects, try converting.
        try:
            new_time = np.array([datetime.datetime(t) for t in time])
        except TypeError:
            try:
                # If cannot be converted simply, elements may be strings
                # Try converting to datetime using sunpy.time.parse_time
                new_time = np.array([parse_time(t) for t in time])
            except:
                # Otherwise raise error telling user to input an array
                # of datetime objects.
                raise TypeError("time must be an array or array-like of "
                                "datetime objects or valid time strings.")
        else:
            raise TypeError("time must be an array or array-like of "
                            "datetime objects or valid time strings.")
    return new_time
