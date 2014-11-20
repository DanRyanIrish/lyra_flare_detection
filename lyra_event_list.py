from __future__ import absolute_import
from __future__ import division

import os.path
from datetime import datetime
from datetime import timedelta
from warnings import warn
import copy
import csv
import urllib

import numpy as np
import sqlite3
import pandas
from itertools import chain
from astropy.io import fits

from sunpy.time import parse_time
from sunpy import config
from sunpy.util import net

import matplotlib.pyplot as plt

RISE_FACTOR = 1.01
FALL_FACTOR = 0.5
# Set mean daily minimum irradiance in Zr channel from first light
# (Jan 2010) until mid 2014.
NORM = 0.001

LYTAF_REMOTE_PATH = "http://proba2.oma.be/lyra/data/lytaf/"
LYTAF_PATH = os.path.expanduser(os.path.join("~", "pro",
                                             "lyra_flare_detection", "data"))
LYRA_DATA_PATH = config.get("downloads", "download_dir")
LYRA_REMOTE_DATA_PATH = "http://proba2.oma.be/lyra/data/bsd"

def generate_lyra_event_list(start_time, end_time, lytaf_path=LYTAF_PATH,
                             exclude_eclipse_season=True):
    """
    Creates a LYRA flare list based on an input start and end time.

    Parameters
    ----------
    

    """
    # Ensure input start and end times are datetime objects
    start_time = parse_time(start_time)
    end_time = parse_time(end_time)
    if start_time >= end_time:
        raise InputError("Input start time after end time.")
    # Create list of datetime objects for each day in time period.
    start_until_end = end_time-start_time
    dates = [start_time+timedelta(days=i) for i in range(start_until_end.days)]
    # Exclude dates during LYRA eclipse season if keyword set and raise
    # warning any dates are skipped.
    if exclude_eclipse_season:
        dates, skipped_dates = _remove_lyra_eclipse_dates(dates)
    # Raise Warning here if dates are skipped
    for date in skipped_dates:
        warn("{0} has been skipped due to LYRA eclipse season.".format(date))
    # Raise Error if no valid dates remain
    if dates == []:
        raise ValueError("No valid dates within input date range.")
    # Define numpy recarray to store lyra event list from csv file
    lyra_events = np.empty((0,), dtype=[("start_time", object),
                                        ("peak_time", object),
                                        ("end_time", object),
                                        ("start_flux", float),
                                        ("peak_flux", float),
                                        ("end_flux", float)])
    # Create list of required FITS files from dates, where consecutive
    # files are contained in a sublist.  Download any not found locally.
    fitsfiles = []
    prev_date = dates[0]
    for date in dates:
        fitsfile = \
          "lyra_{0}-000000_lev3_std.fits".format(date.strftime("%Y%m%d"))
        net.check_download_file(fitsfile,
                                "{0}/{1}".format(LYRA_REMOTE_DATA_PATH,
                                                 date.strftime("%Y/%m/%d/")),
                                LYRA_DATA_PATH)
        if (date-prev_date).days == 1:
            fitsfiles[-1].append(fitsfile)
        else:
            fitsfiles.append([fitsfile])
        prev_date = copy.deepcopy(date)
    for consecutive_files in fitsfiles:
        # Get times and irradiances from first FITS file.
        hdulist = fits.open(os.path.join(LYRA_DATA_PATH, consecutive_files[0]))
        date = parse_time(hdulist[0].header["DATE-OBS"])
        if hdulist[1].header["TUNIT1"] == "MIN":
            t = [date+timedelta(minutes=int(tu))
                 for tu in hdulist[1].data["TIME"]]
        else:
            raise ValueError("Time unit in FITS header not recognised."
                             "  Must be 'MIN'.")
        time = np.asanyarray(t)
        flux = np.asanyarray(hdulist[1].data["CHANNEL4"])
        # If there is no consecutive file before or after this one,
        # apply flare detection to single day
        if len(consecutive_files) == 1:
            # Apply flare detection algorithm
            lyra_events = np.append(lyra_events, find_lyra_events(
                time, flux, lytaf_path=lytaf_path))
        else:
            # Else, if more than on file, apply flare detection
            # algorithm in two day chunks, but iterate by only one day.
            # This simulates the effect of the flare detection algorithm
            # in the PROBA-2/LYRA pipeline being applied every day to
            # the current and previous day.  This will give the same
            # results as the official LYRA flare list.
            prev_time = copy.deepcopy(time)
            prev_flux = copy.deepcopy(flux)
            for fitsfile in consecutive_files[1:]:
                # Delete events in new events from previous day.
                lyra_events = lyra_events[np.logical_not(np.logical_and(
                    lyra_events["start_time"] >= prev_time[0],
                    lyra_events["start_time"] < prev_time[-1]))]
                # Get times and irradiances from FITS file.
                hdulist = fits.open(os.path.join(LYRA_DATA_PATH, fitsfile))
                date = parse_time(hdulist[0].header["DATE-OBS"])
                if hdulist[1].header["TUNIT1"] == "MIN":
                    t = [date+timedelta(minutes=int(tu))
                         for tu in hdulist[1].data["TIME"]]
                else:
                    raise ValueError("Time unit in FITS header not recognised."
                                     "  Must be 'MIN'.")
                # concatenate time and flux from current fits file
                # with that from previous consecutive fits file.
                time = np.append(prev_time, np.asanyarray(t))
                flux = np.append(prev_flux,
                                 np.asanyarray(hdulist[1].data["CHANNEL4"]))
                # Apply flare detection algorithm
                lyra_events = np.append(lyra_events, find_lyra_events(
                    time, flux, lytaf_path=lytaf_path))
                prev_time = np.asanyarray(t)
                prev_flux = np.asanyarray(hdulist[1].data["CHANNEL4"])
    return lyra_events

def find_lyra_events(time, flux, lytaf_path=LYTAF_PATH):
    """
    Finds events in a times series satisfying LYRA event definitions.

    This function finds events/flares in an input time series which satisfy
    the LYRA event definitions and returns the start, peak and end times
    and fluxes.  The LYRA event definitions have been devised for the
    Zirconium channel (channel 4).  For more info, see Notes section of this
    docstring.

    Parameters
    ----------
    flux : ndarray/array-like convertible to float64, e.g. np.array, list
        Contains flux/irradiance measurements
    time : ndarray/array-like of of datetime objects, e.g. np.array, list
        Contains measurement times corresponding to each element in
        flux.  Must be same length as flux.
    lytaf_path : string
        directory path where the LYRA annotation files are stored.

    Returns
    -------
    lyra_events : numpy recarray
        Contains the start, peak and end times and flux values fo each event
        found.  The fields of the recarray are: 'start_time', 'peak_time',
        'end_time', 'start_flux', 'peak_flux', 'end_flux'.

    Notes
    -----
    The LYRA event definitions have been devised for the Zirconium channel
    (channel 4).

    Start time definition:
    1) There must be 4 consecutive minutes when the gradient of the
    1-minute-averaged flux is positive.
    2) The flux in the 4th minute must be at least 1% greater than that in
    the first minute.
    3) The start time is then the earliest consecutive positive gradient before
    the time identified by crieteria 1) and 2).
    N.B. The time series is additively scaled so that the median is
    0.001 W/m^2 (The mean flux for LYRA channel 4 from the start of the
    mission to mid 2014).  Criteria 2) is applied to this scaled data, not the
    observed.  This helps reduce the bias of detections due to variability in
    the solar background flux.

    End time definition:
    1) The flux must fall to half-way between the peak and initial fluxes.
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
    flux = np.asanyarray(flux, dtype="float64")
    time = _check_datetime(time)
    # Define recarray to store results
    lyra_events = np.empty((0,), dtype=[("start_time", object),
                                        ("peak_time", object),
                                        ("end_time", object),
                                        ("start_flux", float),
                                        ("peak_flux", float),
                                        ("end_flux", float)])
    # object LYRA artifacts from timeseries
    clean_time, fluxlist, artifact_status = remove_lyra_artifacts(time, [flux],
        artifacts=["UV occ.", "Offpoint", "LAR", "Calibration", "SAA",
                   "Vis occ.", "Operational Anomaly", "Glitch", "ASIC reload",
                   "Moon in LYRA", "Recovery"], return_artifacts=True,
                   lytaf_path=lytaf_path)
    clean_flux = fluxlist[0]
    artifacts_removed = artifact_status["removed"]
    # Perform subtraction so median irradiance of time series is at
    # average daily minimum from first 4 years of mission.
    clean_flux_scaled = clean_flux - (np.median(clean_flux)-NORM)
    # Get derivative of flux wrt time
    time_timedelta = clean_time[1:-1]-clean_time[0:-2]
    dt = np.zeros(len(time_timedelta), dtype="float64")
    for i, t, in enumerate(time_timedelta):
        dt[i] = t.total_seconds()
    dfdt = np.gradient(clean_flux_scaled[0:-2], dt)
    # Get locations where derivative is positive
    pos_deriv = np.where(dfdt > 0)[0]
    neg_deriv = np.where(dfdt < 0)[0]
    pos_deriv0 = np.where(dfdt >= 0)[0]
    neg_deriv0 = np.where(dfdt <= 0)[0]
    # Find difference between each time point and the one 4
    # observations ahead.
    time_timedelta4 = clean_time[4:-1]-clean_time[0:-5]
    dt4 = np.zeros(len(time_timedelta4))
    for i, t, in enumerate(time_timedelta4):
        dt4[i] = t.total_seconds()
    # Find all possible flare start times.
    end_series = len(clean_flux_scaled)-1
    i=0
    while i < len(pos_deriv)-4:
        # Start time criteria
        if (pos_deriv[i:i+4]-pos_deriv[i] == np.arange(4)).all() and \
          dt4[pos_deriv[i]] > 210 and dt4[pos_deriv[i]] < 270 and \
          clean_flux_scaled[pos_deriv[i+4]]/clean_flux_scaled[pos_deriv[i]] \
          >= RISE_FACTOR:
            # Find start time which is defined as earliest continuous
            # increase in flux before the point found by the above
            # criteria.
            try:
                k = np.where(neg_deriv0 < pos_deriv[i])[0][-1]
                kk = np.where(pos_deriv > neg_deriv0[k])[0][0]
            except IndexError:
                kk = i
            start_index = pos_deriv[kk]
            # If artifact is at start of flare, set start time to
            # directly afterwards.
            artifact_check = np.logical_and(
                artifacts_removed["end_time"] > clean_time[start_index],
                artifacts_removed["end_time"] < clean_time[pos_deriv[kk+2]])
            if artifact_check.any() == True:
                artifact_at_start = artifacts_removed[artifact_check][-1]
                new_index = np.where(
                    clean_time[pos_deriv] > artifact_at_start["end_time"])
                start_index = pos_deriv[new_index[0][0]]
            # Next, find index of flare end time.
            # If flare has not ended, do not record it.
            try:
                jj = np.where(neg_deriv > start_index)[0][0]
            except IndexError:
                i = i+1
            else:
                j = neg_deriv[jj]
                end_condition = False
                while end_condition == False and j < end_series:
                    j = j+1
                    maxflux = max(clean_flux_scaled[start_index:j])
                    end_condition = clean_flux_scaled[j] <= \
                      maxflux-(maxflux-clean_flux_scaled[start_index]) \
                      *FALL_FACTOR
                if j >= end_series:
                    i = i+1
                else:
                    try:
                        m = np.where(pos_deriv0 > j)[0][0]
                    except IndexError:
                        i = i+1
                    else:
                        end_index = pos_deriv0[m]-1
                        # If artifact is at end of flare, set end time
                        # to directly beforehand.
                        artifact_check = np.logical_and(
                            artifacts_removed["begin_time"] < clean_time[end_index],
                            artifacts_removed["begin_time"] > clean_time[end_index-2])
                        if artifact_check.any() == True:
                            artifact_at_end = artifacts_removed[artifact_check][0]
                            new_index = np.where(
                                clean_time < artifact_at_end["begin_time"])
                            end_index = new_index[0][-1]
                        # find index of peak time
                        peak_index = np.where(clean_flux_scaled == \
                            max(clean_flux_scaled[start_index:end_index]))
                        peak_index = peak_index[0][0]
                        # Record flare start, peak and end times
                        lyra_events = np.append(
                            lyra_events, np.empty(1, dtype=lyra_events.dtype))
                        lyra_events[-1]["start_time"] = clean_time[start_index]
                        lyra_events[-1]["peak_time"] = clean_time[peak_index]
                        lyra_events[-1]["end_time"] = clean_time[end_index]
                        lyra_events[-1]["start_flux"] = clean_flux[start_index]
                        lyra_events[-1]["peak_flux"] = clean_flux[peak_index]
                        lyra_events[-1]["end_flux"] = clean_flux[end_index]
                        # If the most recently found flare is during the
                        # decay phase of another reset end time of
                        # previous flare to start time of this flare.
                        if len(lyra_events) > 1 and \
                          lyra_events[-2]["end_time"] > lyra_events[-1]["start_time"]:
                            lyra_events[-2]["end_time"] = lyra_events[-1]["start_time"]
                            lyra_events[-2]["end_flux"] = lyra_events[-1]["start_flux"]
                        # Finally, set principle iterator, i, to the
                        # peak of the flare just found so that algorithm
                        # will start looking for flares during the decay
                        # phase of this flare and beyond.  This ensures
                        # that flares during the decay phase are also
                        # located.
                        i = np.where(pos_deriv > peak_index)[0][0]
        else:
            i = i+1

    return lyra_events

def remove_lyra_artifacts(time, fluxes=None, artifacts="All",
                          return_artifacts=False, fitsfile=None,
                          csvfile=None, filecolumns=None,
                          lytaf_path=LYTAF_PATH):
    """
    Removes periods of LYRA artifacts from a time series.

    This functions removes periods corresponding to certain artifacts recorded
    in the LYRA annotation file from an array of times given by the time input.
    If a list of arrays of other properties is supplied through the fluxes
    kwarg, then the relevant values from these arrays are also removed.  This
    is done by assuming that each element in each array supplied corresponds to
    the time in the same index in time array.  The artifacts to be removed are
    given via the artifacts kwarg.  The default is "all", meaning that all
    artifacts will be removed.  However, a subset of artifacts can be removed
    by supplying a list of strings of the desired artifact types.

    Parameters
    ----------
    time : ndarray/array-like of datetime objects
        Gives the times of the timeseries.

    fluxes : (optional) list of ndarrays/array-likes convertible to float64.
        Contains the fluxes/properties taken at the times in the time array.
        Each element in the list must have the same number of elements as time.

    artifacts : list of strings
        Contain the artifact types to be removed.  For list of artifact types
        see reference [1].  For example, if user wants to remove only large
        angle rotations, listed at reference [1] as LAR, let artifacts=["LAR"].
        Default='All', i.e. all artifacts will be removed.

    return_artifacts : (optional) bool
        Set to True to return a numpy recarray containing the start time, end
        time and type of all artifacts removed.
        Default=False

    fitsfile : (optional) string
        file name (including file path and suffix, .fits) of output fits file
        which is generated if this kwarg is not None.
        Default=None, i.e. no fits file is output.

    csvfile : (optional) string
        file name (including file path and suffix, .csv) of output csv file
        which is generated if this kwarg is not None.
        Default=None, i.e. no csv file is output.

    filecolumns : (optional) list of strings
        Gives names of columns of any output files produced.  Although
        initially set to None above, the default is in fact
        ["time", "flux0", "flux1",..."fluxN"]
        where N is the number of flux arrays in the fluxes input
        (assuming 0-indexed counting).

    lytaf_path : string
        directory path where the LYRA annotation files are stored.
        
    Returns
    -------
    clean_time : ndarray/array-like of datetime objects
        time array with artifact periods removed.

    clean_fluxes : (optional) list ndarrays/array-likes convertible to float64
        list of fluxes with artifact periods removed.

    artifact_status : (optional) dictionary
        List of 4 variables containing information on what artifacts were
        found, removed, etc. from the time series.
        artifact_status["lytaf"] = artifacts found : numpy recarray
            The full LYRA annotation file for the time series time range
            output by extract_combined_lytaf().
        artifact_status["removed"] = artifacts removed : numpy recarray
            Artifacts which were found and removed from from time series.
        artifact_status["not_removed"] = artifacts found but not removed :
              numpy recarray
            Artifacts which were found but not removed as they were not
            included when user defined artifacts kwarg.
        artifact_status["not_found"] = artifacts not found : list of strings
            Artifacts listed to be removed by user when defining artifacts
            kwarg which were not found in time series time range.

    References
    ----------
    [1] http://proba2.oma.be/data/TARDIS

    Example
    -------

    """
    # Check inputs
    time = _check_datetime(time)
    if not all(isinstance(artifact_type, str) for artifact_type in artifacts):
        raise TypeError("All elements in artifacts must in strings.")
    if type(fluxes) is not None and type(fluxes) is not list:
        raise TypeError("fluxes must be None or a list of numpy arrays of "
                        "dtype 'float64'.")
    # Define outputs
    clean_time = copy.deepcopy(time)
    clean_fluxes = copy.deepcopy(fluxes)
    artifacts_not_found =[]
    # Get LYTAF file for given time range
    lytaf = extract_combined_lytaf(time[0], time[-1], lytaf_path=lytaf_path)
    
    # Find events in lytaf which are to be removed from time series.
    if artifacts == "All":
        artifact_indices = np.arange(len(lytaf["begin_time"]))
    else:
        artifact_indices = np.empty(0, dtype="int64")
        for artifact_type in artifacts:
            indices = np.where(lytaf["event_type"] == artifact_type)[0]
            # If none of a given type of artifact is found, record this
            # type in artifact_not_found list.
            if len(indices) == 0:
                artifacts_not_found.append(artifact_type)
            else:
                # Else, record the indices of the artifacts of this type
                artifact_indices = np.concatenate((artifact_indices, indices))
        artifact_indices.sort()

    # Remove relevant artifacts from timeseries. If none of the
    # artifacts the user wanted removed were found, raise a warning and
    # continue with code.
    if len(artifact_indices) == 0:
        warn("None of user supplied artifacts were found.")
        artifacts_not_found = artifacts
    else:
        # Remove periods corresponding to artifacts from flux and time
        # arrays.
        bad_indices = np.empty(0, dtype="int64")
        all_indices = np.arange(len(time))
        for index in artifact_indices:
            bad_period = np.logical_and(time >= lytaf["begin_time"][index],
                                        time <= lytaf["end_time"][index])
            bad_indices = np.append(bad_indices, all_indices[bad_period])
        clean_time = np.delete(time, bad_indices)
        if fluxes is not None:
            for i, f in enumerate(clean_fluxes):
                clean_fluxes[i] = np.delete(f, bad_indices)
    # If return_artifacts kwarg is True, return a list containing
    # information on what artifacts found, removed, etc.  See docstring.
    if return_artifacts is True:
        if artifacts_not_found == artifacts:
            artifact_status = {"lytaf": lytaf,
                               "removed": lytaf[artifact_indices],
                               "not_removed": None,
                               "not_found": artifacts_not_found}
        else:
            artifacts_removed = lytaf[artifact_indices]
            artifacts_not_removed = np.delete(lytaf, artifact_indices)
            if artifacts == "All":
                artifacts_not_found = None
            artifact_status = {"lytaf": lytaf, "removed": artifacts_removed,
                               "not_removed": artifacts_not_removed,
                               "not_found": artifacts_not_found}
    # Output FITS file if fits kwarg is set
    if fitsfile != None:
        # Create time array of time strings rather than datetime objects
        # and verify filecolumns have been correctly input.  If None,
        # generate generic filecolumns (see docstring og function called
        # below.
        string_time, filecolumns = _prep_columns(time, fluxes, filecolumns)
        # Prepare column objects.
        cols = [fits.Column(name=filecolumns[0], format="26A",
                            array=string_time)]
        if fluxes != None:
            for i, f in enumerate(fluxes):
                cols.append(fits.Column(name=filecolumns[i+1], format="D",
                                        array=f))
        coldefs = fits.ColDefs(cols)
        tbhdu = fits.new_table(coldefs)
        hdu = fits.PrimaryHDU()
        tbhdulist = fits.HDUList([hdu, tbhdu])
        # Write data to fits file.
        tbhdulist.writeto(fitsfile)
    # Output csv file if fits kwarg is set.
    if csvfile != None:
        # Create time array of time strings rather than datetime objects
        # and verify filecolumns have been correctly input.  If None,
        # generate generic filecolumns (see docstring og function called
        # below.
        string_time, filecolumns = prep_columns(time, fluxes, filecolumns)
        # Open and write data to csv file.
        with open(csvfile, 'w') as openfile:
            csvwriter = csv.writer(openfile, delimiter=';')
            # Write header.
            csvwriter.writerow(filecolumns)
            # Write data.
            if fluxes == None:
                for i in range(len(time)):
                    csvwriter.writerow(string_time[i])
            else:
                for i in range(len(time)):
                    row = [string_time[i]]
                    for f in fluxes:
                        row.append(f[i])
                    csvwriter.writerow(row)
    # Return values.
    if return_artifacts is True:
        if fluxes is None:
            return clean_time, artifact_status
        else:
            return clean_time, clean_fluxes, artifact_status
    else:
        if fluxes is None:
            return clean_time
        else:
            return clean_time, clean_fluxes

def extract_combined_lytaf(start_time, end_time, lytaf_path=LYTAF_PATH,
                           combine_files=["lyra", "manual", "ppt", "science"],
                           csvfile=None):
    """
    Extracts combined lytaf file for given time range.

    Given a time range defined by start_time and end_time, this function
    extracts the segments of each LYRA annotation file and combines them.

    Parameters
    ----------
    start_time : datetime object or string
        Start time of period for which annotation file is required.
    end_time : datetime object or string
        End time of period for which annotation file is required.
    lytaf_path : string
        directory path where the LYRA annotation files are stored.
    combine_files : (optional) list of strings
        States which LYRA annotation files are to be combined.
        Default is all four, i.e. lyra, manual, ppt, science.
        See Notes section for an explanation of each.

    Returns
    -------
    lytaf : numpy record array containing the various parameters stored
        in the LYTAF files.

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
    
    """
    # Check inputs
    # Check start_time is a date string or datetime object
    if type(start_time) is str:
        start_time = parse_time(start_time)
    if type(start_time) is not datetime:
        raise TypeError("start_time must be a date string or datetime object")
    # Check start_time is a date string or datetime object
    if type(end_time) is str:
        end_time = parse_time(end_time)
    if type(end_time) is not datetime:
        raise TypeError("end_time must be a date string or datetime object")
    # Check combine_files contains correct inputs
    if not all(suffix in ["lyra", "manual", "ppt", "science"]
               for suffix in combine_files):
        raise TypeError("Elements in combine_files must be strings equalling "
                        "'lyra', 'manual', 'ppt', or 'science'.")
    # Remove any duplicates from combine_files input
    combine_files = list(set(combine_files))
    combine_files.sort()
    # Convert input times to UNIX timestamp format since this is the
    # time format in the annotation files
    start_time_uts = (start_time - datetime(1970, 1, 1)).total_seconds()
    end_time_uts = (end_time - datetime(1970, 1, 1)).total_seconds()

    # Define numpy record array which will hold the information from
    # the annotation file.
    lytaf = np.empty((0,), dtype=[("insertion_time", object),
                               ("begin_time", object),
                               ("reference_time", object),
                               ("end_time", object),
                               ("event_type", object),
                               ("event_definition", object)])
    # Access annotation files
    for i, suffix in enumerate(combine_files):
        # Check database files are present
        dbname = "annotation_{0}.db".format(suffix)
        net.check_download_file(dbname, LYTAF_REMOTE_PATH, lytaf_path)
        # Open SQLITE3 annotation files
        connection = sqlite3.connect(os.path.join(lytaf_path, dbname))
        # Create cursor to manipulate data in annotation file
        cursor = connection.cursor()
        # Check if lytaf file spans the start and end times defined by
        # user.  If not, download newest version.
        # First get start time of first event and end time of last event in lytaf.
        cursor.execute("select begin_time from event order by begin_time asc "
                       "limit 1;")
        db_first_begin_time = cursor.fetchone()[0]
        db_first_begin_time = datetime.fromtimestamp(db_first_begin_time)
        cursor.execute("select end_time from event order by end_time desc "
                       "limit 1;")
        db_last_end_time = cursor.fetchone()[0]
        db_last_end_time = datetime.fromtimestamp(db_last_end_time)
        # If lytaf does not include entire input time range...
        if end_time > db_last_end_time or start_time < db_first_begin_time:
            # ...close lytaf file...
            cursor.close()
            connection.close()
            # ...Download latest lytaf file...
            net.check_download_file(dbname, LYTAF_REMOTE_PATH, lytaf_path,
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
        # Enter desired information into the lytaf numpy record array
        for event_row in event_rows:
            id_index = eventType_id.index(event_row[4])
            lytaf = np.append(
                lytaf, np.array((datetime.utcfromtimestamp(event_row[0]),
                                 datetime.utcfromtimestamp(event_row[1]),
                                 datetime.utcfromtimestamp(event_row[2]),
                                 datetime.utcfromtimestamp(event_row[3]),
                                 eventType_type[id_index],
                                 eventType_definition[id_index]),
                                 dtype=lytaf.dtype))
        # Close file
        cursor.close()
        connection.close()
    # Sort lytaf in ascending order of begin time
    np.recarray.sort(lytaf, order="begin_time")

    # If csvfile kwarg is set, write out lytaf to csv file
    if csvfile != None:
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

    #return event_rows, eventType_rows
    return lytaf

def _check_datetime(time):
    """
    Checks or tries to convert input array to array of datetime objects.

    Returns input time array with elements as datetime objects or raises an
    TypeError if time not of valid format.  Input format can be anything
    convertible to datetime by datetime() function or any time string valid as
    an input to sunpy.time.parse_time().

    """
    if (np.array([type(t) for t in time]) == datetime).all():
        new_time = np.asanyarray(time)
    elif type(time) == pandas.tseries.index.DatetimeIndex:
        new_time = time.to_pydatetime()        
    else:
        # If elements of time are not datetime objects, try converting.
        try:
            new_time = np.array([datetime(t) for t in time])
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

def _remove_lyra_eclipse_dates(dates):
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

def testing_find_lyra_events(find_events=False):
    fitspath = "../data/LYRA/fits/"
    #fitsname = "lyra_20100201-000000_lev3_std.fits"
    #fitsname = "lyra_20100301-000000_lev3_std.fits"
    #fitsname = "lyra_20100309-000000_lev3_std.fits"
    #fitsname = "lyra_20100401-000000_lev3_std.fits"
    #fitsname = "lyra_20100501-000000_lev3_std.fits"
    #fitsname = "lyra_20100601-000000_lev3_std.fits"
    #fitsname = "lyra_20100701-000000_lev3_std.fits"
    #fitsname = "lyra_20100801-000000_lev3_std.fits"
    #fitsname = "lyra_20100901-000000_lev3_std.fits"
    #fitsname = "lyra_20101001-000000_lev3_std.fits"
    #fitsname = "lyra_20101101-000000_lev3_std.fits"
    #fitsname = "lyra_20101201-000000_lev3_std.fits"
    #fitsname = "lyra_20110101-000000_lev3_std.fits"
    #fitsname = "lyra_20110201-000000_lev3_std.fits"
    #fitsname = "lyra_20110301-000000_lev3_std.fits"
    #fitsname = "lyra_20110401-000000_lev3_std.fits"
    #fitsname = "lyra_20110501-000000_lev3_std.fits"
    #fitsname = "lyra_20110601-000000_lev3_std.fits"
    #fitsname = "lyra_20110701-000000_lev3_std.fits"
    #fitsname = "lyra_20110801-000000_lev3_std.fits"
    #fitsname = "lyra_20110901-000000_lev3_std.fits"
    #fitsname = "lyra_20111001-000000_lev3_std.fits"
    #fitsname = "lyra_20111101-000000_lev3_std.fits"
    #fitsname = "lyra_20111201-000000_lev3_std.fits"
    #fitsname = "lyra_20120101-000000_lev3_std.fits"
    #fitsname = "lyra_20120201-000000_lev3_std.fits"
    #fitsname = "lyra_20120301-000000_lev3_std.fits"
    #fitsname = "lyra_20120401-000000_lev3_std.fits"
    #fitsname = "lyra_20120501-000000_lev3_std.fits"
    #fitsname = "lyra_20120601-000000_lev3_std.fits"
    #fitsname = "lyra_20120701-000000_lev3_std.fits"
    #fitsname = "lyra_20120801-000000_lev3_std.fits"
    #fitsname = "lyra_20120901-000000_lev3_std.fits"
    #fitsname = "lyra_20121001-000000_lev3_std.fits"
    #fitsname = "lyra_20121101-000000_lev3_std.fits"
    #fitsname = "lyra_20121201-000000_lev3_std.fits"
    #fitsname = "lyra_20130101-000000_lev3_std.fits"
    #fitsname = "lyra_20130201-000000_lev3_std.fits"
    #fitsname = "lyra_20130301-000000_lev3_std.fits"
    #fitsname = "lyra_20130401-000000_lev3_std.fits"
    #fitsname = "lyra_20130501-000000_lev3_std.fits"
    #fitsname = "lyra_20130601-000000_lev3_std.fits"
    #fitsname = "lyra_20130701-000000_lev3_std.fits"
    #fitsname = "lyra_20130801-000000_lev3_std.fits"
    #fitsname = "lyra_20130901-000000_lev3_std.fits"
    #fitsname = "lyra_20131001-000000_lev3_std.fits"
    #fitsname = "lyra_20131101-000000_lev3_std.fits"
    #fitsname = "lyra_20131201-000000_lev3_std.fits"
    fitsname = "lyra_20100505-000000_lev3_std.fits"
    fitsfile = fitspath + fitsname
    ly = fits.open(fitsfile)
    t = ly[1].data["TIME"]
    orig_flux = ly[1].data["CHANNEL4"]
    orig_time = np.empty(len(t), dtype="object")
    for i, tt in enumerate(t):
        orig_time[i] = datetime(int(fitsname[5:9]), int(fitsname[9:11]),
                                int(fitsname[11:13]), 0, 0) + \
                                timedelta(0,0,0,0,int(tt))
    time = copy.deepcopy(orig_time)
    flux = copy.deepcopy(orig_flux)
    time, flux, artifacts = remove_lyra_artifacts(time, [flux],
        artifacts=["UV occ.", "Offpoint", "LAR", "Calibration", "SAA",
                   "Vis occ.", "Operational Anomaly", "Glitch", "ASIC reload",
                   "Recovery"], return_artifacts=True)
    flux = flux[0]
    if find_events is False:
        return orig_time, orig_flux, time, flux, artifacts
    else:
        ev = find_lyra_events(time, flux)
        ind = []
        start_ind = []
        end_ind = []
        for i in range(len(ev)):
            ind.append(np.where(time==ev[i]["start_time"])[0][0])
            ind.append(np.where(time==ev[i]["end_time"])[0][0])
            start_ind.append(np.where(time==ev[i]["start_time"])[0][0])
            end_ind.append(np.where(time==ev[i]["end_time"])[0][0])
        plt.ion()
        plt.plot(time, flux, 'o', color='blue')
        plt.plot(time[start_ind], flux[start_ind], 'o', color='green')
        plt.plot(time[end_ind], flux[end_ind], 'o', color='red')
        return orig_time, orig_flux, time, flux, artifacts, ev, ind

def test_cal():
    lla = extract_combined_lytaf("2010-03-01", "2014-07-01",
                                 combine_files=["lyra"])
    ical = np.where(lla["event_type"] == u'Calibration')[0]
    fitspath = "../data/LYRA/fits/"
    i=0
    st = lla["end_time"][ical[i]]
    lytaf = extract_combined_lytaf(
        "{0}-{1}-{2} 00:00".format(
            st.year, st.strftime('%m'), st.strftime('%d')),
            "{0}-{1}-{2} 23:59".format(
            st.year, st.strftime('%m'), st.strftime('%d')))
    fitsname = "lyra_{0}{1}{2}-000000_lev3_std.fits".format(
        st.year, st.strftime('%m'), st.strftime('%d'))
    fitsfile = fitspath + fitsname
    urllib.urlretrieve(
        "http://proba2.oma.be/lyra/data/bsd/{0}/{1}/{2}/{3}".format(
            st.year, st.strftime('%m'), st.strftime('%d'), fitsname), fitsfile)
    ly = fits.open(fitsfile)
    flux = ly[1].data["CHANNEL4"]
    t = ly[1].data["TIME"]
    time = np.empty(len(t), dtype="object")
    for j, tt in enumerate(t):
        time[j] = datetime(int(fitsname[5:9]), int(fitsname[9:11]),
                           int(fitsname[11:13]), 0, 0) + \
                           timedelta(0,0,0,0,int(tt))
    clean_time, fluxlist = remove_lyra_artifacts(time, [flux],
        artifacts=["UV occ.", "Offpoint", "LAR", "Calibration", "SAA",
                   "Vis occ.", "Operational Anomaly", "Glitch", "ASIC reload",
                   "Moon in LYRA"])
    clean_flux = fluxlist[0]
    ly.close()
    print lla["begin_time"][ical[i]], lla["end_time"][ical[i]]
    plt.ion()
    plt.plot(clean_time, clean_flux, 'o')
    plt.title("{0}/{1}/{2}".format(clean_time[0].year, clean_time[0].month,
                                   clean_time[0].day))
    return lytaf
