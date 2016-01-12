from __future__ import absolute_import
from __future__ import division

import os.path
import datetime
from warnings import warn
import copy
import csv
import sqlite3
from urllib2 import HTTPError
import urlparse
import inspect

import numpy as np
import pandas.tseries.index
from itertools import chain
from astropy.io import fits
from sunpy import config
from sunpy.time import parse_time
from sunpy.util.net import check_download_file

RISE_FACTOR = 1.01
FALL_FACTOR = 0.5
FIRST_FLARE_FALL_FACTOR = 0.2

LYTAF_PATH = os.path.expanduser(os.path.join("~", "pro", "data", "LYRA", "LYTAF"))
LYTAF_REMOTE_PATH = "http://proba2.oma.be/lyra/data/lytaf/"
LYRA_DATA_PATH = os.path.expanduser(os.path.join("~", "pro", "data", "LYRA", "fits"))
LYRA_REMOTE_DATA_PATH = "http://proba2.oma.be/lyra/data/bsd/"

ARTIFACTS = ["UV occ.", "Offpoint", "LAR", "SAA", "Calibration", "ASIC reload",
             "Vis. occ.", "Glitch", "Operational Anomaly", "Moon in LYRA",
             "Recovery"]

def generate_lyra_event_list(start_time, end_time, lytaf_path=LYTAF_PATH,
                             exclude_occultation_season=True):
    """
    Generates a LYRA flare list without rescaling data.

    Parameters
    ----------
    start_time : time format compatible by sunpy.time.parse_time()
        start time of period for flare list to be generated.

    end_time : time format compatible by sunpy.time.parse_time()
        end time of period for flare list to be generated.

    lytaf_path : string
        directory path where the LYRA annotation files are stored.

    exclude_eclipse_season : bool
        Determines whether LYRA UV Occulation season is discarded from
        input period.  Default=True.

    Returns
    -------
    new_lyra_flares : numpy recarray
        Each row holds infomation on a flare found by the LYRA flare
        detection algorithm.  Each column holds the following information.
        ["start_time"] : start time of the flare.  datetime object
        ["peak_time"] : peak time of the flare.  datetime object
        ["end_time"] : end time of the flare.  datetime object
        ["start_irrad"] : irradiance value at start time of flare.  float
        ["peak_irrad"] : irradiance value at peak time of flare.  float
        ["end_irrad"] : irradiance value at end time of flare.  float
        ["end_time_comment"] : comment explaining which criterion
            triggered end time.  str

    Notes
    -----
    This function calls a version of find_lyra_flares() to detect flares.  The
    data is not rescaled to remove effect of background variation.For the flare
    definitions, see the Notes in the docstring of find_lyra_flares(),
    below.

    In addition, the LYRA flare list is currently not run during the LYRA
    occulation season during which LYRA periodically passes behind the
    Earth (mid-September to mid-March).  This is because the artifacts in
    the data can not currently be removed and cause find_lyra_flares() to
    be unreliable.  Therefore, by default the kwarg
    exclude_occultation_season=True which ensures that the algorithm is
    not run during these months, even if they are included in the time
    period input by the user.  It does not affect other parts of the time
    period input by the user.  This can be disabled by setting the
    exclude_occultation_season=False.  However, the results produced
    during the occultation season will not be reliable.  It is currently
    highly recommended that the exclude_occultation_season kwarg is left
    at True.

    Examples
    --------
    To reproduce the LYRA flare list for 2010
        >>> lyra_flares_2010 = generate_lyra_flare_list("2010-01-01", "2011-01-01")

    """
    # Produce time series for input dates.
    data = create_lyra_time_series(
        start_time, end_time, level=3, channels=[4], lytaf_path=lytaf_path,
        exclude_occultation_season=exclude_occultation_season)
    # Find flares by calling find_lyra_flares
    lyra_flares = find_lyra_flares(data["TIME"], data["CHANNEL4"], lytaf_path=lytaf_path)
    return lyra_flares

def create_lyra_time_series(start_time, end_time, level=3, channels=[1,2,3,4],
                            lytaf_path=LYTAF_PATH,
                            exclude_occultation_season=False):
    """
    Creates a time series of LYRA standard Unit 2 data.

    Parameters
    ----------
    start_time : time format compatible by sunpy.time.parse_time()
        start time of period for flare list to be generated.

    end_time : time format compatible by sunpy.time.parse_time()
        end time of period for flare list to be generated.

    level : `int` equal to 1, 2, or 3.
        LYRA data level.
        1: raw data
        2: calibrated data
        3: one minuted-averaged data

    channels : `list` of ints in range 1-4 inclusive.
        Channels for which time series are to be created.
        1: Lyman Alpha
        2: Herzberg
        3: Aluminium
        4: Zirconium

    lytaf_path : string
        directory path where the LYRA annotation files are stored.

    exclude_eclipse_season : bool
        Determines whether LYRA UV Occulation season is discarded from
        input period.  Default=True.

    Returns
    -------
    data : `numpy.recarray`
        Time series for input period containing time and irradiance
        values for each channel.

    Examples
    --------

    """
    # Check that inputs are correct.
    if not level in range(1,4):
        raise ValueError("level must be an int equal to 1, 2, or 3. " + \
                         "Value entered = {0}".format(level))
    if not all(channels) in range(1,5):
        raise ValueError("Values in channels must be ints equal to 1, 2, 3, or 4." + \
                         "Value entered = {0}".format(level))
    # Ensure input start and end times are datetime objects
    start_time = parse_time(start_time)
    end_time = parse_time(end_time)
    # Create list of datetime objects for each day in time period.
    start_until_end = end_time-start_time
    dates = [start_time+datetime.timedelta(days=i)
             for i in range(start_until_end.days+1)]
    # Exclude dates during LYRA eclipse season if keyword set and raise
    # warning any dates are skipped.
    if exclude_occultation_season:
        dates, skipped_dates = _remove_lyra_occultation_dates(dates)
        # Raise Warning here if dates are skipped
        for date in skipped_dates:
            warn("{0} has been skipped due to LYRA eclipse season.".format(date))
    # Raise Error if no valid dates remain
    if dates == []:
        raise ValueError("No valid dates within input date range.")
    # Search for daily FITS files for input time period.
    # First, create empty arrays to hold entire time series of input
    # time range.
    data_dtypes = [("CHANNEL{0}".format(channel), float) for channel in channels]
    data_dtypes.insert(0, ("TIME", object))
    data = np.empty((0,), dtype=data_dtypes)
    for date in dates:
        fitsfile = "lyra_{0}-000000_lev{1}_std.fits".format(
            date.strftime("%Y%m%d"), level)
        # Check each fitsfile exists locally.  If not, download it.
        try:
            check_download_file(fitsfile,
                                "{0}/{1}".format(LYRA_REMOTE_DATA_PATH,
                                                 date.strftime("%Y/%m/%d/")),
                                LYRA_DATA_PATH)
            # Append data in files to time series
            with fits.open(os.path.join(LYRA_DATA_PATH, fitsfile)) as hdulist:
                n = len(hdulist[1].data)
                data = np.append(data, np.empty((n,), dtype=data_dtypes))
                data["TIME"][-n:] = _time_list_from_lyra_fits(hdulist)
                for channel in channels:
                    data["CHANNEL{0}".format(channel)][-n:] = \
                      hdulist[1].data["CHANNEL{0}".format(channel)]
        except HTTPError:
            warn("Skipping file as it could not be found: {0}".format(urlparse.urljoin(
                "{0}{1}".format(LYRA_REMOTE_DATA_PATH,
                                date.strftime("%Y/%m/%d/")), fitsfile)))
        # Truncate time series to match start and end input times.
        w = np.logical_and(data["TIME"] >= start_time, data["TIME"] < end_time)
        data = data[w]
    return data
    

def _generate_lyra_event_list_scaled(start_time, end_time, lytaf_path=LYTAF_PATH,
                                    exclude_occultation_season=True):
    """
    Generates a LYRA flare list in the same way as in the LYRA data pipeline.

    Parameters
    ----------
    start_time : time format compatible by sunpy.time.parse_time()
        start time of period for flare list to be generated.

    end_time : time format compatible by sunpy.time.parse_time()
        end time of period for flare list to be generated.

    lytaf_path : string
        directory path where the LYRA annotation files are stored.

    exclude_eclipse_season : bool
        Determines whether LYRA UV Occulation season is discarded from
        input period.  Default=True.

    Returns
    -------
    new_lyra_flares : numpy recarray
        Each row holds infomation on a flare found by the LYRA flare
        detection algorithm.  Each column holds the following information.
        ["start_time"] : start time of the flare.  datetime object
        ["peak_time"] : peak time of the flare.  datetime object
        ["end_time"] : end time of the flare.  datetime object
        ["start_irrad"] : irradiance value at start time of flare.  float
        ["peak_irrad"] : irradiance value at peak time of flare.  float
        ["end_irrad"] : irradiance value at end time of flare.  float

    Notes
    -----
    This function calls find_lyra_flares() to detect flares.  For the flare
    definitions, see the Notes in the docstring of find_lyra_flares(),
    below.  find_lyra_flares() rescales the median of the data to a common
    level so as to reduce the effect of background variation on which flares
    are detected and missed over long time scales.  Therefore, the time
    period over which the algorithm is run can have an effect on the
    results.  In order to reproduce the LYRA flare list create by the LYRA
    data pipeline, this function takes the time period input by the user
    and applies find_lyra_flares() iteratively over two day intervals with
    an iteration of one day.  This happens in the LYRA data pipeline for
    operational reasons.  It means that many days are often processed
    twice with a small chance of the detected flares changing slightly
    between the two runs.  In this case it is the 2nd set of detections
    that are recorded.  Although this is inefficient, it is reproduces the
    results of the LYRA flare list produced in the data pipeline. The
    exception is when no data is available before or after a given day
    due to data gaps.  In this case, that day's data is searched in isolation.

    In addition, the LYRA flare list is currently not run during the LYRA
    occulation season during which LYRA periodically passes behind the
    Earth (mid-September to mid-March).  This is because the artifacts in
    the data can not currently be removed and cause find_lyra_flares() to
    be unreliable.  Therefore, by default the kwarg
    exclude_occultation_season=True which ensures that the algorithm is
    not run during these months, even if they are included in the time
    period input by the user.  It does not affect other parts of the time
    period input by the user.  This can be disabled by setting the
    exclude_occultation_season=False.  However, the results produced
    during the occultation season will not be reliable.  It is currently
    highly recommended that the exclude_occultation_season kwarg is left
    at True.

    Examples
    --------
    To reproduce the LYRA flare list for 2010
        >>> lyra_flares_2010 = generate_lyra_flare_list("2010-01-01", "2011-01-01")

    """
    # Ensure input start and end times are datetime objects
    start_time = parse_time(start_time)
    end_time = parse_time(end_time)
    # Create list of datetime objects for each day in time period.
    start_until_end = end_time-start_time
    dates = [start_time+datetime.timedelta(days=i) for i in range(start_until_end.days)]
    # Exclude dates during LYRA eclipse season if keyword set and raise
    # warning any dates are skipped.
    if exclude_occultation_season:
        dates, skipped_dates = _remove_lyra_occultation_dates(dates)
        # Raise Warning here if dates are skipped
        for date in skipped_dates:
            warn("{0} has been skipped due to LYRA eclipse season.".format(date))
    # Raise Error if no valid dates remain
    if dates == []:
        raise ValueError("No valid dates within input date range.")
    # Define numpy recarray to store lyra event list from csv file
    new_lyra_flares = np.empty((0,), dtype=[("start_time", object),
                                            ("peak_time", object),
                                            ("end_time", object),
                                            ("start_irrad", float),
                                            ("peak_irrad", float),
                                            ("end_irrad", float),
                                            ("end_time_comment", "a40")])
    # Create list of required FITS files from dates, where consecutive
    # files are contained in a sublist.  Download any not found locally.
    fitsfiles = []
    prev_date = dates[0]
    for date in dates:
        fitsfile = \
          "lyra_{0}-000000_lev3_std.fits".format(date.strftime("%Y%m%d"))
        # Check each fitsfile exists locally.  If not, download it.
        try:
            check_download_file(fitsfile,
                                "{0}/{1}".format(LYRA_REMOTE_DATA_PATH,
                                                 date.strftime("%Y/%m/%d/")),
                                LYRA_DATA_PATH)
            # Make sure there are data after artifacts removed
            hdulist = fits.open(os.path.join(LYRA_DATA_PATH, fitsfile))
            clean_time, irradiance_list = remove_lytaf_events(
                np.asanyarray(_time_list_from_lyra_fits(hdulist)),
                [np.asanyarray(hdulist[1].data["CHANNEL4"])],
                artifacts=ARTIFACTS, lytaf_path=lytaf_path)
            if len(clean_time) != 0:
                # Create sublists in fitsfiles list where each each
                # sublist only contains fitsfiles for consecutive days.
                # If a fitsfile does not exist or contains no good
                # data, start a new sublist.
                if (date-prev_date).days == 1:
                    fitsfiles[-1].append(os.path.join(LYRA_DATA_PATH, fitsfile))
                else:
                    fitsfiles.append([os.path.join(LYRA_DATA_PATH, fitsfile)])
                prev_date = copy.deepcopy(date)
        except HTTPError:
            warn("Could not find {0}".format(urlparse.urljoin(
                "{0}{1}".format(LYRA_REMOTE_DATA_PATH, date.strftime("%Y/%m/%d/")),
                fitsfile)))
    # Perform flare detection on consecutive days
    for consecutive_files in fitsfiles:
        # When there are more than one consecutive fitsfile,
        # perform flare detection on current and previous day.
        if len(consecutive_files) > 1:
            # Extract data from first fits file.
            hdulist = fits.open(consecutive_files[0])
            if hdulist[1].header["TUNIT1"] == "MIN":
                prev_time = np.asanyarray(_time_list_from_lyra_fits(hdulist))
            else:
                raise ValueError("Time unit in FITS file not recognised.  "
                                 "Must be 'MIN': {0}".format(consecutive_files[0]))
            prev_irradiance = np.asanyarray(hdulist[1].data["CHANNEL4"])
            prev_date = parse_time(hdulist[0].header["DATE-OBS"])
            for i in range(len(consecutive_files[1:])):
                # Extract data from next file.
                hdulist = fits.open(consecutive_files[i+1])
                if hdulist[1].header["TUNIT1"] == "MIN":
                    current_time = np.asanyarray(_time_list_from_lyra_fits(hdulist))
                else:
                    raise ValueError("Time unit in FITS file not recognised.  "
                                     "Must be 'MIN': {0}".format(consecutive_files[0]))
                current_irradiance = np.asanyarray(hdulist[1].data["CHANNEL4"])
                date = parse_time(hdulist[0].header["DATE-OBS"])
                time = np.concatenate((prev_time, current_time))
                irradiance = np.concatenate((prev_irradiance, current_irradiance))
                # Find lyra flares
                print "Seeking flares on {0} -- {1}".format(
                    prev_date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"))
                some_new_lyra_flares = find_lyra_flares(time, irradiance,
                                                        lytaf_path=lytaf_path)
                # Delete events in new events from previous day.
                new_lyra_flares = new_lyra_flares[np.logical_not(
                    np.logical_and(
                        new_lyra_flares["start_time"] >= prev_time[0],
                        new_lyra_flares["start_time"] < prev_time[-1]))]
                # Append events found in this iteration.
                new_lyra_flares = np.append(new_lyra_flares, some_new_lyra_flares)
                # Set previous time and irradiance arrays to current
                # day's values for next iteration.
                prev_time = copy.deepcopy(current_time)
                prev_irradiance = copy.deepcopy(current_irradiance)
                prev_date = copy.deepcopy(date)
        # When previous and next file are not consecutive...
        # perform flare detection algorithm on single day.
        else:
            # Extract data from first fits file.
            hdulist = fits.open(consecutive_files[0])
            if hdulist[1].header["TUNIT1"] == "MIN":
                time = np.asanyarray(_time_list_from_lyra_fits(hdulist))
            else:
                raise ValueError("Time unit in FITS file not recognised.  "
                                 "Must be 'MIN': {0}".format(consecutive_files[0]))
            irradiance = np.asanyarray(hdulist[1].data["CHANNEL4"])
            # Find lyra flares and write out results to csv file
            print "Seeking flares on {0}".format(date.strftime("%Y-%m-%d"))
            nf = find_lyra_flares(time, irradiance, lytaf_path=lytaf_path)
            new_lyra_flares = np.concatenate((new_lyra_flares, nf))
    return new_lyra_flares

def find_lyra_flares(time, irradiance, lytaf_path=LYTAF_PATH):
    """
    Finds events in a times series satisfying LYRA event definitions.

    This function finds events/flares in an input time series which satisfy
    the LYRA event definitions and returns the start, peak and end times
    and channels.  The LYRA event definitions have been devised for the
    Zirconium channel (channel 4).  For more info, see Notes section of this
    docstring.

    Parameters
    ----------
    irradiance : ndarray/array-like convertible to float64, e.g. np.array, list
        Contains irradiance measurements
    time : ndarray/array-like of of datetime objects, e.g. np.array, list
        Contains measurement times corresponding to each element in
        irradiance.  Must be same length as irradiance.
    lytaf_path : string
        directory path where the LYRA annotation files are stored.

    Returns
    -------
    lyra_flares : numpy recarray
        Contains the start, peak and end times and irradiance values for each
        event found.  The fields of the recarray are: 'start_time',
        'peak_time', 'end_time', 'start_irrad', 'peak_irrad', 'end_irrad'.

    Notes
    -----
    The LYRA event definitions have been devised for the Zirconium channel
    (channel 4).

    Start time definition:
    1) There must be 4 consecutive minutes when the gradient of the
    1-minute-averaged irradiance is positive.
    2) The irradiance in the 4th minute must be at least 1% greater than that
    in the first minute.
    3) The start time is then the earliest consecutive positive gradient
    before the time identified by crieteria 1) and 2).
    N.B. The time series is additively scaled so that the median is
    0.001 W/m^2 (The mean irradiance for LYRA channel 4 from the start of the
    mission to mid 2014).  Criteria 2) is applied to this scaled data, not the
    observed.  This helps reduce the bias of detections due to variability in
    the solar background irradiance.

    End time definition:
    1) The irradiance must fall to half-way between the peak and initial
    irradiances.
    2) The end time is  then the latest consecutive negative gradient after
    the time identifie by criterion 1).

    Artifacts:
    The following artifacts, as defined in the LYRA time annotation file,
    (see reference [1]) are removed from the time series:
    "UV occ.", "Offpoint", "LAR", "Calibration", "SAA", "Vis occ.",
    "Operational Anomaly", "Glitch", "ASIC reload", "Moon in LYRA", "Recovery".
    In some cases this may cause a flare start or end times to be recorded
    slightly differently to what the eye would intuitively identify in the
    full data.  It may also cause some flares not be detected at all.
    However, it also drastically cuts down on false detections.

    References
    ---------
    [1] http://proba2.oma.be/data/TARDIS

    Examples
    --------

    """
    # Ensure inputs are of correct type
    irradiance = np.asanyarray(irradiance, dtype="float64")
    time = _check_datetime(time)
    # object LYRA artifacts from timeseries
    clean_time, irradiance_list, artifact_status = remove_lytaf_events(
        time, [irradiance], artifacts=ARTIFACTS, return_artifacts=True,
        lytaf_path=lytaf_path)
    clean_irradiance = irradiance_list[0]
    # Remove data points <= 0
    w = np.logical_and(clean_irradiance > 0., clean_irradiance < 10.)
    clean_time = clean_time[w]
    clean_irradiance = clean_irradiance[w]
    # Apply LYRAFF
    lyra_flares = apply_lyraff(clean_time, clean_irradiance, 
                               artifacts_removed=artifact_status["removed"])

    return lyra_flares

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


def remove_lytaf_events(time, channels=None, artifacts=None,
                        return_artifacts=False, fitsfile=None,
                        csvfile=None, filecolumns=None,
                        lytaf_path=None, force_use_local_lytaf=False):
    """
    Removes periods of LYRA artifacts from a time series.

    This functions removes periods corresponding to certain artifacts recorded
    in the LYRA annotation file from an array of times given by the time input.
    If a list of arrays of other properties is supplied through the channels
    kwarg, then the relevant values from these arrays are also removed.  This
    is done by assuming that each element in each array supplied corresponds to
    the time in the same index in time array.  The artifacts to be removed are
    given via the artifacts kwarg.  The default is "all", meaning that all
    artifacts will be removed.  However, a subset of artifacts can be removed
    by supplying a list of strings of the desired artifact types.

    Parameters
    ----------
    time : `numpy.ndarray` of `datetime.datetime`
        Gives the times of the timeseries.

    channels : `list` of `numpy.array` convertible to float64.
        Contains arrays of the irradiances taken at the times in the time
        variable.  Each element in the list must have the same number of
        elements as time.

    artifacts : `list` of strings
        Contain the artifact types to be removed.  For list of artifact types
        see reference [1].  For example, if user wants to remove only large
        angle rotations, listed at reference [1] as LAR, let artifacts=["LAR"].
        Default=[], i.e. no artifacts will be removed.

    return_artifacts : `bool`
        Set to True to return a numpy recarray containing the start time, end
        time and type of all artifacts removed.
        Default=False

    fitsfile : `str`
        file name (including file path and suffix, .fits) of output fits file
        which is generated if this kwarg is not None.
        Default=None, i.e. no fits file is output.

    csvfile : `str`
        file name (including file path and suffix, .csv) of output csv file
        which is generated if this kwarg is not None.
        Default=None, i.e. no csv file is output.

    filecolumns : `list` of strings
        Gives names of columns of any output files produced.  Although
        initially set to None above, the default is in fact
        ["time", "channel0", "channel1",..."channelN"]
        where N is the number of irradiance arrays in the channels input
        (assuming 0-indexed counting).

    lytaf_path : `str`
        directory path where the LYRA annotation files are stored.

    force_use_local_lytaf : `bool`
        Ensures current local version of lytaf files are not replaced by
        up-to-date online versions even if current local lytaf files do not
        cover entire input time range etc.
        Default=False

    Returns
    -------
    clean_time : `numpy.ndarray` of `datetime.datetime`
        time array with artifact periods removed.

    clean_channels : `list` ndarrays/array-likes convertible to float64
        list of irradiance arrays with artifact periods removed.

    artifact_status : `dict`
        List of 4 variables containing information on what artifacts were
        found, removed, etc. from the time series.
        artifact_status["lytaf"] = artifacts found : `numpy.recarray`
            The full LYRA annotation file for the time series time range
            output by get_lytaf_events().
        artifact_status["removed"] = artifacts removed : `numpy.recarray`
            Artifacts which were found and removed from from time series.
        artifact_status["not_removed"] = artifacts found but not removed :
              `numpy.recarray`
            Artifacts which were found but not removed as they were not
            included when user defined artifacts kwarg.
        artifact_status["not_found"] = artifacts not found : `list` of strings
            Artifacts listed to be removed by user when defining artifacts
            kwarg which were not found in time series time range.

    References
    ----------
    [1] http://proba2.oma.be/data/TARDIS

    Example
    -------
    Sample data for example
        >>> time = np.array([datetime(2013, 2, 1)+timedelta(minutes=i)
                             for i in range(120)])
        >>> channel_1 = np.zeros(len(TIME))+0.4
        >>> channel_2 = np.zeros(len(TIME))+0.1
    Remove LARs (Large Angle Rotations) from time series.
        >>> time_clean, channels_clean = remove_lyra_artifacts(
              time, channels=[channel_1, channel2], artifacts=['LAR'])

    """
    # Check inputs
    if not lytaf_path:
        lytaf_path = LYTAF_PATH
    if channels and type(channels) is not list:
        raise TypeError("channels must be None or a list of numpy arrays "
                        "of dtype 'float64'.")
    if not artifacts:
        raise ValueError("User has supplied no artifacts to remove.")
    if type(artifacts) is str:
      artifacts = [artifacts]
    if not all(isinstance(artifact_type, str) for artifact_type in artifacts):
        raise TypeError("All elements in artifacts must in strings.")
    all_lytaf_event_types = get_lytaf_event_types(lytaf_path=lytaf_path,
                                                  print_event_types=False)
    for artifact in artifacts:
        if not artifact in all_lytaf_event_types:
            print all_lytaf_event_types
            raise ValueError("{0} is not a valid artifact type.".format(artifact))
    # Define outputs
    clean_time = np.array([parse_time(t) for t in time])
    clean_channels = copy.deepcopy(channels)
    artifacts_not_found = []
    # Get LYTAF file for given time range
    print "Getting LYTAF events."
    lytaf = get_lytaf_events(time[0], time[-1], lytaf_path=lytaf_path,
                             force_use_local_lytaf=force_use_local_lytaf)
    print "Got LYTAF events."

    # Find events in lytaf which are to be removed from time series.
    artifact_indices = np.empty(0, dtype="int64")
    for artifact_type in artifacts:
        print "Seeking indices of {0}".format(artifact_type)
        indices = np.where(lytaf["event_type"] == artifact_type)[0]
        # If none of a given type of artifact is found, record this
        # type in artifact_not_found list.
        if len(indices) == 0:
            artifacts_not_found.append(artifact_type)
        else:
            # Else, record the indices of the artifacts of this type
            artifact_indices = np.concatenate((artifact_indices, indices))
        print "Found indices of {0}".format(artifact_type)
    artifact_indices.sort()

    # Remove relevant artifacts from timeseries. If none of the
    # artifacts the user wanted removed were found, raise a warning and
    # continue with code.
    if not len(artifact_indices):
        warn("None of user supplied artifacts were found.")
        artifacts_not_found = artifacts
    else:
        # Remove periods corresponding to artifacts from flux and time
        # arrays.
        bad_indices = np.empty(0, dtype="int64")
        all_indices = np.arange(len(time))
        nn = len(artifact_indices)
        for index in artifact_indices:
            bad_period = np.logical_and(time >= lytaf["begin_time"][index],
                                        time <= lytaf["end_time"][index])
            bad_indices = np.append(bad_indices, all_indices[bad_period])
        clean_time = np.delete(clean_time, bad_indices)
        if channels:
            for i, f in enumerate(clean_channels):
                clean_channels[i] = np.delete(f, bad_indices)
    # If return_artifacts kwarg is True, return a list containing
    # information on what artifacts found, removed, etc.  See docstring.
    if return_artifacts:
        artifact_status = {"lytaf": lytaf,
                           "removed": lytaf[artifact_indices],
                           "not_removed": np.delete(lytaf, artifact_indices),
                           "not_found": artifacts_not_found}
    # Output FITS file if fits kwarg is set
    if fitsfile:
        # Create time array of time strings rather than datetime objects
        # and verify filecolumns have been correctly input.  If None,
        # generate generic filecolumns (see docstring of function called
        # below.
        string_time, filecolumns = _prep_columns(time, channels, filecolumns)
        # Prepare column objects.
        cols = [fits.Column(name=filecolumns[0], format="26A",
                            array=string_time)]
        if channels:
            for i, f in enumerate(channels):
                cols.append(fits.Column(name=filecolumns[i+1], format="D",
                                        array=f))
        coldefs = fits.ColDefs(cols)
        tbhdu = fits.new_table(coldefs)
        hdu = fits.PrimaryHDU()
        tbhdulist = fits.HDUList([hdu, tbhdu])
        # Write data to fits file.
        tbhdulist.writeto(fitsfile)
    # Output csv file if csv kwarg is set.
    if csvfile:
        # Create time array of time strings rather than datetime objects
        # and verify filecolumns have been correctly input.  If None,
        # generate generic filecolumns (see docstring of function called
        # below.
        string_time, filecolumns = _prep_columns(time, channels, filecolumns)
        # Open and write data to csv file.
        with open(csvfile, 'w') as openfile:
            csvwriter = csv.writer(openfile, delimiter=';')
            # Write header.
            csvwriter.writerow(filecolumns)
            # Write data.
            if not channels:
                for i in range(len(time)):
                    csvwriter.writerow(string_time[i])
            else:
                for i in range(len(time)):
                    row = [string_time[i]]
                    for f in channels:
                        row.append(f[i])
                    csvwriter.writerow(row)
    # Return values.
    if return_artifacts:
        if not channels:
            return clean_time, artifact_status
        else:
            return clean_time, clean_channels, artifact_status
    else:
        if not channels:
            return clean_time
        else:
            return clean_time, clean_channels

def get_lytaf_events(start_time, end_time, lytaf_path=None,
                     combine_files=("lyra", "manual", "ppt", "science"),
                     csvfile=None, force_use_local_lytaf=False):
    """
    Extracts combined lytaf file for given time range.

    Given a time range defined by start_time and end_time, this function
    extracts the segments of each LYRA annotation file and combines them.

    Parameters
    ----------
    start_time : `datetime.datetime` or `str`
        Start time of period for which annotation file is required.

    end_time : `datetime.datetime` or `str`
        End time of period for which annotation file is required.

    lytaf_path : `str`
        directory path where the LYRA annotation files are stored.

    combine_files : `tuple` of strings
        States which LYRA annotation files are to be combined.
        Default is all four, i.e. lyra, manual, ppt, science.
        See Notes section for an explanation of each.

    force_use_local_lytaf : `bool`
        Ensures current local version of lytaf files are not replaced by
        up-to-date online versions even if current local lytaf files do not
        cover entire input time range etc.
        Default=False

    Returns
    -------
    lytaf : `numpy.recarray`
        Containsing the various parameters stored in the LYTAF files.

    Notes
    -----
    There are four LYRA annotation files which mark different types of events
    or artifacts in the data.  They are named annotation_suffix.db where
    suffix is a variable equalling either lyra, manual, ppt, or science.

    annotation_lyra.db : contains entries regarding possible effects to
        the data due to normal operation of LYRA instrument.

    annotation_manual.db : contains entries regarding possible effects
        to the data due to unusual or manually logged events.

    annotation_ppt.db : contains entries regarding possible effects to
        the data due to pointing or positioning of PROBA2.

    annotation_science.db : contains events in the data scientifically
        interesting, e.g. GOES flares.

    References
    ----------
    Further documentation: http://proba2.oma.be/data/TARDIS

    Examples
    --------
    Get all events in the LYTAF files for January 2014
        >>> lytaf = get_lytaf_events('2014-01-01', '2014-02-01')

    """
    # Check inputs
    # Check lytaf path
    if not lytaf_path:
        lytaf_path = LYTAF_PATH
    # Check start_time and end_time is a date string or datetime object
    start_time = parse_time(start_time)
    end_time = parse_time(end_time)
    # Check combine_files contains correct inputs
    if not all(suffix in ["lyra", "manual", "ppt", "science"]
               for suffix in combine_files):
        raise ValueError("Elements in combine_files must be strings equalling "
                         "'lyra', 'manual', 'ppt', or 'science'.")
    # Remove any duplicates from combine_files input
    combine_files = list(set(combine_files))
    combine_files.sort()
    # Convert input times to UNIX timestamp format since this is the
    # time format in the annotation files
    start_time_uts = (start_time - datetime.datetime(1970, 1, 1)).total_seconds()
    end_time_uts = (end_time - datetime.datetime(1970, 1, 1)).total_seconds()

    # Define numpy record array which will hold the information from
    # the annotation file.
    lytaf = np.empty((0,), dtype=[("insertion_time", object),
                                  ("begin_time", object),
                                  ("reference_time", object),
                                  ("end_time", object),
                                  ("event_type", object),
                                  ("event_definition", object)])
    # Access annotation files
    for suffix in combine_files:
        print "Accessing {0} database.".format(suffix)
        # Check database files are present
        dbname = "annotation_{0}.db".format(suffix)
        check_download_file(dbname, LYTAF_REMOTE_PATH, lytaf_path)
        # Open SQLITE3 annotation files
        connection = sqlite3.connect(os.path.join(lytaf_path, dbname))
        # Create cursor to manipulate data in annotation file
        cursor = connection.cursor()
        # Check if lytaf file spans the start and end times defined by
        # user.  If not, download newest version.
        # First get start time of first event and end time of last
        # event in lytaf.
        cursor.execute("select begin_time from event order by begin_time asc "
                       "limit 1;")
        db_first_begin_time = cursor.fetchone()[0]
        db_first_begin_time = datetime.datetime.fromtimestamp(db_first_begin_time)
        cursor.execute("select end_time from event order by end_time desc "
                       "limit 1;")
        db_last_end_time = cursor.fetchone()[0]
        db_last_end_time = datetime.datetime.fromtimestamp(db_last_end_time)
        # If lytaf does not include entire input time range...
        if not force_use_local_lytaf:
            if end_time > db_last_end_time or start_time < db_first_begin_time:
                # ...close lytaf file...
                cursor.close()
                connection.close()
                # ...Download latest lytaf file...
                check_download_file(dbname, LYTAF_REMOTE_PATH, lytaf_path,
                                    replace=True)
                # ...and open new version of lytaf database.
                connection = sqlite3.connect(os.path.join(lytaf_path, dbname))
                cursor = connection.cursor()
        # Select and extract the data from event table within file within
        # given time range
        cursor.execute("select insertion_time, begin_time, reference_time, "
                       "end_time, eventType_id from event where end_time >= "
                       "{0} and begin_time <= "
                       "{1}".format(start_time_uts, end_time_uts))
        event_rows = cursor.fetchall()
        # Select and extract the event types from eventType table
        cursor.row_factory = sqlite3.Row
        cursor.execute("select * from eventType")
        eventType_rows = cursor.fetchall()
        eventType_id = []
        eventType_type = []
        eventType_definition = []
        for eventType_row in eventType_rows:
            eventType_id.append(eventType_row["id"])
            eventType_type.append(eventType_row["type"])
            eventType_definition.append(eventType_row["definition"])
        print "Entering {0} entries into recarray.".format(suffix)
        # Enter desired information into the lytaf numpy record array
        mm = len(event_rows)
        ii=-1
        for event_row in event_rows:
            ii=ii+1
            id_index = eventType_id.index(event_row[4])
            lytaf = np.append(lytaf,
                              np.array((datetime.datetime.utcfromtimestamp(event_row[0]),
                                        datetime.datetime.utcfromtimestamp(event_row[1]),
                                        datetime.datetime.utcfromtimestamp(event_row[2]),
                                        datetime.datetime.utcfromtimestamp(event_row[3]),
                                        eventType_type[id_index],
                                        eventType_definition[id_index]), dtype=lytaf.dtype))
        # Close file
        cursor.close()
        connection.close()
    # Sort lytaf in ascending order of begin time
    np.recarray.sort(lytaf, order="begin_time")

    # If csvfile kwarg is set, write out lytaf to csv file
    if csvfile:
        # Open and write data to csv file.
        with open(csvfile, 'w') as openfile:
            csvwriter = csv.writer(openfile, delimiter=';')
            # Write header.
            csvwriter.writerow(lytaf.dtype.names)
            # Write data.
            for row in lytaf:
                new_row = []
                new_row.append(row[0].strftime("%Y-%m-%dT%H:%M:%S"))
                new_row.append(row[1].strftime("%Y-%m-%dT%H:%M:%S"))
                new_row.append(row[2].strftime("%Y-%m-%dT%H:%M:%S"))
                new_row.append(row[3].strftime("%Y-%m-%dT%H:%M:%S"))
                new_row.append(row[4])
                new_row.append(row[5])
                csvwriter.writerow(new_row)

    return lytaf

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

def get_lytaf_event_types(lytaf_path=None, print_event_types=True):
    """Prints the different event types in the each of the LYTAF databases.

    Parameters
    ----------
    lytaf_path : `str`
        Path location where LYTAF files are stored.
        Default = LYTAF_PATH defined above.

    print_event_types : `bool`
        If True, prints the artifacts in each lytaf database to screen.

    Returns
    -------
    all_event_types : `list`
        List of all events types in all lytaf databases.

    """
    # Set lytaf_path is not done by user
    if not lytaf_path:
        lytaf_path = LYTAF_PATH
    suffixes = ["lyra", "manual", "ppt", "science"]
    all_event_types = []
    # For each database file extract the event types and print them.
    if print_event_types:
        print "\nLYTAF Event Types\n-----------------\n"
    for suffix in suffixes:
        dbname = "annotation_{0}.db".format(suffix)
        # Check database file exists, else download it.
        check_download_file(dbname, LYTAF_REMOTE_PATH, lytaf_path)
        # Open SQLITE3 LYTAF files
        connection = sqlite3.connect(os.path.join(lytaf_path, dbname))
        # Create cursor to manipulate data in annotation file
        cursor = connection.cursor()
        cursor.execute("select type from eventType;")
        event_types = cursor.fetchall()
        all_event_types.append(event_types)
        if print_event_types:
            print "----------------\n{0} database\n----------------".format(suffix)
            for event_type in event_types:
                print str(event_type[0])
            print " "
    # Unpack event types in all_event_types into single list
    all_event_types = [event_type[0] for event_types in all_event_types
                       for event_type in event_types]
    return all_event_types

def _prep_columns(time, fluxes, filecolumns):
    """
    Checks and prepares data to be written out to a file.

    Firstly, this function converts the elements of time, whose entries are
    assumed to be datetime objects, to time strings.  Secondly, it checks
    whether the number of elements in an input list of columns names,
    filecolumns, is equal to the number of arrays in the list, fluxes.  If not,
    a Value Error is raised.  If however filecolumns equals None, a filenames
    list is generated equal to ["time", "fluxes0", "fluxes1",...,"fluxesN"]
    where N is the number of arrays in the list, fluxes
    (assuming 0-indexed counting).

    """
    # Convert time which contains datetime objects to time strings.
    string_time = np.empty(len(time), dtype="S26")
    for i, t in enumerate(time):
        string_time[i] = t.strftime("%Y-%m-%dT%H:%M:%S.%f")

    # If filenames is given...
    if filecolumns != None:
        # ...check all the elements are strings...
        if all(isinstance(column, str) for column in filecolumns) is False:
            raise TypeError("All elements in filecolumns must by strings.")
        # ...and that there are the same number of elements as there
        # are arrays in fluxes, plus 1 for a time array.  Otherwise
        # raise a ValueError.
        if fluxes != None:
            ncol = 1 + len(fluxes)
        else:
            ncol = 1
        if len(filecolumns) != ncol:
            raise ValueError("Number of elements in filecolumns must be "
                             "equal to the number of input data arrays, "
                             "i.e. time + elements in fluxes.")
    # If filenames not given, create a list of columns names of the
    # form: ["time", "fluxes0", "fluxes1",...,"fluxesN"] where N is the
    # number of arrays in fluxes (assuming 0-indexed counting).
    else:
        if fluxes != None:
            filecolumns = ["flux{0}".format(fluxnum)
                           for fluxnum in range(len(fluxes))]
            filecolumns.insert(0, "time")
        else:
            filecolumns = ["time"]

    return string_time, filecolumns

def _timedelta_totalseconds(timedelta_obj):
    """Manually implements timedelta.total_seconds() method.

    Computes total seconds manually to be compatible with Python 2.6.
    See https://docs.python.org/2/library/datetime.html
    """
    return (timedelta_obj.microseconds + (timedelta_obj.seconds +
                       timedelta_obj.days*24*3600) * 10**6) / 10**6

def _remove_lyra_occultation_dates(dates):
    """Given a list of datetimes, removes those during LYRA eclipse season."""
    non_eclipse_dates = [date for date in dates if date.month != 10 and
                         date.month != 11 and date.month != 12 and
                         date.month != 1 and date.month != 2]
    non_eclipse_dates = [date for date in non_eclipse_dates
                         if not (date.month == 9 and date.day > 15)]
    non_eclipse_dates = [date for date in non_eclipse_dates
                         if not (date.month == 3 and date.day < 15)]
    non_eclipse_dates.sort()
    # Determine skipped dates
    skipped_dates = list(set(dates) - set(non_eclipse_dates))
    skipped_dates.sort()
    return non_eclipse_dates, skipped_dates

def _time_list_from_lyra_fits(hdulist):
    """Generates a list of measurement times from LYRA FITS header."""
    if hdulist[1].header["TUNIT1"] == "MIN":
        #start_time = datetime.strptime(hdulist[0].header["DATE"], "%Y-%m-%d")
        start_time = parse_time(hdulist[0].header["DATE-OBS"])
        start_time = datetime.datetime(start_time.year, start_time.month,
                              start_time.day)
        if hdulist[1].header["TUNIT1"] == "MIN":
            t = [start_time+datetime.timedelta(minutes=int(tu))
                 for tu in hdulist[1].data["TIME"]]
        elif hdulist[1].header["TUNIT1"] == "s":
            t = [start_time+datetime.timedelta(seconds=int(tu))
                 for tu in hdulist[1].data["TIME"]]
        else:
            raise ValueError(
                "Time unit in fits file not recognised.  Should be 'MIN'.")
    return t
