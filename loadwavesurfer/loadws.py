from collections import namedtuple
from datetime import datetime
from scipy import signal

import numpy as np
import pandas as pd
from pywavesurfer import ws
from pathlib import Path

Data = namedtuple("Data", ("volt", "curr", "time", "ttl"))
Params = namedtuple(
    "Params",
    (
        "timestamps",
        "delays",
        "pulseCounts",
        "pulseDurations",
        "delaysBetweenPulses",
        "ampChangesPerPulse",
        "firstPulseAmps",
        "stimLengths",
    ),
)


class LoadWaveSurfer:

    def __init__(self, inputFile, filt=False, downsampling_factor=1):
        """Load WaveSurfer data and make it easy to parse

        Parameters
        ----------
        inputFile : str
            Full path to WaveSurfer-generated .h5 files
        filt : bool, optional
            Choose whether the signal should be lowpass and notch filtered, by default False
        downsampling_factor : int, optional
            Downsampling factor to use with `signal.decimate()`, by default 1.0 (no downsampling).        
        """
        if isinstance(inputFile, str):
            if inputFile[-3:] != ".h5":
                inputFile = inputFile + ".h5"
            self.inputFile = Path(inputFile)
        elif isinstance(inputFile, Path):
            self.inputFile = inputFile
        else:
            raise TypeError("Input file must be a string or pathlib.Path object")

        self.filter = filt

        self.ds_factor = downsampling_factor

        self.data()

    def data(self):
        """Load and preprocess WaveSurfer data

        Returns
        -------
        volt : array
            Array with voltage values
        curr : array
            Array with current values
        time : array
            Array with elapsed time
        """
        f = self.rawData()

        # Read recording parameters and calibration data
        self.params = f["header"]
        freq = self.sampleRate()

        # get the time at which "record" was  pressed
        # there is a delay between start of recording and the actual sweep start time
        # so we want to capture that offset in the `time` structure
        clock_at_start = self.clock()

        # Read data
        sweeps = list(f.keys())[1:]

        Volt = []
        Curr = []
        Time = []
        TTL = []

        for sweep in sweeps:
            voltage = f[sweep]["analogScans"][0]
            current = f[sweep]["analogScans"][1]

            time_offset = f[sweep]["timestamp"]
            sweep_start = clock_at_start + time_offset
            time_vec = np.arange(sweep_start, (len(voltage) / freq) + sweep_start, 1 / freq)

            # sometimes our reconstructed time vector may have an extra item in it
            if len(voltage) != len(time_vec):
                time_vec = time_vec[:-1]

            Volt.append(voltage)
            Curr.append(current)
            Time.append(time_vec)

            try:
                ttl = f[sweep]["digitalScans"][0]
                TTL.append(ttl)
            except KeyError as e:
                # No digital signal found in sweep. Silently handle error.
                pass

        self.Volt = np.hstack(Volt)
        self.Curr = np.hstack(Curr)
        self.Time = np.hstack(Time)

        try:
            self.TTL = np.hstack(TTL)
        except ValueError as e:
            self.TTL = [np.nan] * len(self.Volt)
            pass

        # notch and lowpass filter
        if self.filter:
            self.filterData()

        if self.ds_factor > 1:
            self.Time = self.Time[::self.ds_factor]
            self.Volt = signal.decimate(self.Volt, q=self.ds_factor, ftype="fir").flatten()
            self.Curr = signal.decimate(self.Curr, q=self.ds_factor, ftype="fir").flatten()

            try:
                self.TTL = self.TTL[::self.ds_factor]
            except ValueError as e:
                self.TTL = 0
                pass

        return Data(self.Volt, self.Curr, self.Time, self.TTL)

    def filterData(self):
        """Filter raw data to eliminate 60Hz noise and high, non-biological frequencies
        """

        b, a = signal.butter(N=1, Wn=[59, 61], btype="bandstop", output="ba", fs=20e3)
        filtvolt = signal.filtfilt(b, a, self.Volt)
        filtcurr = signal.filtfilt(b, a, self.Curr)

        b, a = signal.butter(N=1, Wn=200, btype="lowpass", output="ba", fs=20e3)
        self.Volt = signal.filtfilt(b, a, filtvolt)
        self.Curr = signal.filtfilt(b, a, filtcurr)

        return

    def rawData(self):
        """Load raw data

        Returns
        -------
        dict
            Dictionary with all the raw data
        """
        return ws.loadDataFile(filename=self.inputFile, format_string="double")

    def toDF(self):
        """Convert WaveSurfer data to pandas DataFrame format
        """
        all_data = {'time': self.Time, 'volt': self.Volt, 'curr': self.Curr, 'ttl': self.TTL}
        df = pd.DataFrame.from_dict(all_data)

        return df

    def stimParams(self):
        """Get all stimulation parameters from the given recording.

        Returns
        -------
        namedtuple
            Named tuple containing `timestamp`, `delay`, `pulseCount`, `pulseDuration`, `delayBetweenPulses`,
            `ampChangePerPulse`, `firstPulseAmp`, `stimLength`, and `realLength`
            - `timestamp`: real start time of stimulation after pressing 'record', in seconds
            - `delay`: predetermined stimulation delay, prepended before any amplitude change
            - `pulseCount`: number of pulses
            - `pulseDuration`: duration of each pulse, in seconds
            - `delayBetweenPulses`: delay between each pulse, if using a train, in seconds
            - `ampChangePerPulse`: amplitude change for each pulse
            - `firstPulseAmp`: amplitude of first pulse
            - `stimLength`: length of stimulation, in seconds
        """

        pulseCountCorrection = False
        if self.inputFile.parent.stem == "20220602":
            # on 20220602 I altered the stimulation library mid-experiment, fixing a mistake
            # in the amount of pulses injected from 12 to 13, so now we have to account for that.
            # Leaving this here to serve as a lesson.
            if "p4" not in self.inputFile.stem:
                pulseCountCorrection = True

        if self.isStimEnabled():
            stim_lib = self.stimLibrary()
            selected_stim = stim_lib["SelectedOutputableClassName"].astype("str")

            if selected_stim == "ws.StimulusMap":
                # Because only one electrode is being used, each stimulus map corresponds to
                # a single stimulus. If a stimulus map is selected, we have it easy.

                map_idx = int(stim_lib["SelectedOutputableIndex"].flatten()[0])
                map_metadata = stim_lib["Maps"][f"element{map_idx}"]
                multiplier = map_metadata["Multiplier"].flatten()[0]

                stim_index = int(stim_lib["SelectedOutputableIndex"].flatten()[0])
                stim_metadata = stim_lib["Stimuli"][f"element{stim_index}"]["Delegate"]

                sweepNumber = self.session().split("_")[-1]

                if stim_metadata["TypeString"].decode() == "SquarePulseLadder":

                    delay = float(stim_metadata["Delay"])
                    timestamp = [self.rawData()[f"sweep_{sweepNumber}"]["timestamp"]]
                    pulseCount = int(stim_metadata["PulseCount"])
                    pulseDuration = float(stim_metadata["PulseDuration"])
                    delayBetweenPulses = float(stim_metadata["DelayBetweenPulses"])
                    ampChangePerPulse = (float(stim_metadata["AmplitudeChangePerPulse"]) *
                                         multiplier)
                    firstPulseAmp = float(stim_metadata["FirstPulseAmplitude"]) * multiplier

                    length = (2 * delay + (pulseDuration + delayBetweenPulses) * pulseCount -
                              delayBetweenPulses)

                elif stim_metadata["TypeString"].decode() == "SquarePulse":
                    delay = float(stim_metadata["Delay"])
                    timestamp = [self.rawData()[f"sweep_{sweepNumber}"]["timestamp"]]
                    pulseCount = 1
                    pulseDuration = float(stim_metadata["Duration"])
                    delayBetweenPulses = 0
                    ampChangePerPulse = 0
                    firstPulseAmp = float(stim_metadata["Amplitude"]) * multiplier

                    length = (2 * delay + (pulseDuration + delayBetweenPulses) * pulseCount -
                              delayBetweenPulses)

                else:
                    print(f"{stim_metadata['TypeString']} has not been implemented yet")
                    return None

                if length > map_metadata["Duration"]:
                    # sometimes, a user (like myself, oops) may incorrectly input a total map duration
                    # that's smaller than the minimum required stimulus duration, so we have to
                    # correct for that. For example, if a given stimulus should last 40 seconds but
                    # the stimulus map duration is overriden during the recording to last 38 seconds,
                    # the appropriate reconstruction should reflect the ground truth data (38s of
                    # stimulation) and not the actual hard-coded length (40s)
                    stimLength = map_metadata["Duration"]
                    pulseCount -= 1

                elif length < self.getSweepDuration():
                    # other times, the stimulation may contain timepoints where the command is
                    # set to 0pA from the last stimulus until the end of the sweep
                    stimLength = self.getSweepDuration()

                else:
                    stimLength = length

                # realLength = stimLength + timestamp[0]

            if selected_stim == "ws.StimulusSequence":
                sequence = int(stim_lib["SelectedOutputableIndex"])
                maps = stim_lib["Sequences"][f"element{sequence}"]["IndexOfEachMapInLibrary"]

                # realLength = []
                timestamp = []

                delay = []
                pulseCount = []
                pulseDuration = []
                delayBetweenPulses = []
                ampChangePerPulse = []
                firstPulseAmp = []
                stimLength = []

                sweepNumber = self.session().split("_")[-1]
                if '-' in sweepNumber and len(sweepNumber) > 5:
                    sweepNumber = sweepNumber.split("-")
                    firstIdx = [i for i, e in enumerate(list(sweepNumber[0])) if e != '0']
                    lastIdx = [i for i, e in enumerate(list(sweepNumber[-1])) if e != '0']

                    sweepNumber = [sweepNumber[0][firstIdx[0]:], sweepNumber[-1][lastIdx[0]:]]

                    sweepNumber = np.arange(int(sweepNumber[0]), int(sweepNumber[-1]) + 1)
                elif '-' in sweepNumber and len(sweepNumber) == 5:
                    sweepNumber = sweepNumber.split("-")[0]

                for count, val in enumerate(maps.values()):
                    map_idx = int(val)
                    
                    if isinstance(sweepNumber, np.ndarray):
                        timestamp.append(
                            self.rawData()[f"sweep_{int(sweepNumber[count]):04}"]["timestamp"])
                    elif isinstance(sweepNumber, str):
                        timestamp.append(
                            self.rawData()[f"sweep_{int(sweepNumber):04}"]["timestamp"])
                        
                    if map_idx != 0:
                        map_metadata = stim_lib["Maps"][f"element{map_idx}"]

                        if isinstance(map_metadata["IndexOfEachStimulusInLibrary"], bytes):
                            break

                        else:
                            multiplier = map_metadata["Multiplier"].flatten()[0]

                            stim_index = int(map_metadata["IndexOfEachStimulusInLibrary"]["element1"])
                            stim_metadata = stim_lib["Stimuli"][f"element{stim_index}"]["Delegate"]

                            stim_type = stim_metadata["TypeString"].astype("str")

                            if stim_type == "SquarePulseLadder":

                                _delay = float(stim_metadata["Delay"])
                                _pulseCount = int(stim_metadata["PulseCount"])
                                _pulseDuration = float(stim_metadata["PulseDuration"])
                                _delayBetweenPulses = float(stim_metadata["DelayBetweenPulses"])
                                _ampChangePerPulse = float(stim_metadata["AmplitudeChangePerPulse"]) * multiplier
                                _firstPulseAmp = float(stim_metadata["FirstPulseAmplitude"]) * multiplier

                                delay.append(_delay)
                                pulseCount.append(_pulseCount)
                                pulseDuration.append(_pulseDuration)
                                delayBetweenPulses.append(_delayBetweenPulses)
                                ampChangePerPulse.append(_ampChangePerPulse)
                                firstPulseAmp.append(_firstPulseAmp)

                                length = (2 * _delay + (_pulseDuration + _delayBetweenPulses) * _pulseCount - _delayBetweenPulses)

                                if pulseCountCorrection:
                                    pulseCount[count] -= 1

                                if length < self.getSweepDuration():
                                    stimLength.append(self.getSweepDuration())

                                else:
                                    stimLength.append(length)

                                # realLength.append(stimLength[count] + timestamp[count] - np.sum(realLength))

                            elif stim_type == "SquarePulse":
                                delay.append(float(stim_metadata["Delay"]))
                                pulseCount.append(1)
                                pulseDuration.append(float(stim_metadata["Duration"]))
                                delayBetweenPulses.append(0)
                                ampChangePerPulse.append(0)
                                firstPulseAmp.append(float(stim_metadata["Amplitude"]) * multiplier)

                                length = 2 * delay[count] + pulseDuration[count]

                                if length > map_metadata["Duration"]:
                                    stimLength.append(map_metadata["Duration"])
                                    pulseCount[count] -= 1

                                elif length < self.getSweepDuration():
                                    stimLength.append(self.getSweepDuration())

                                else:
                                    stimLength.append(length)

                                # realLength.append(stimLength[count] + timestamp[count])
                    elif map_idx == 0:
                        _delay = 0
                        _pulseCount = 0
                        _pulseDuration = 0
                        _delayBetweenPulses = 0
                        _ampChangePerPulse = 0
                        _firstPulseAmp = 0

                        delay.append(_delay)
                        pulseCount.append(_pulseCount)
                        pulseDuration.append(_pulseDuration)
                        delayBetweenPulses.append(_delayBetweenPulses)
                        ampChangePerPulse.append(_ampChangePerPulse)
                        firstPulseAmp.append(_firstPulseAmp)

                        length = 60

        else:
            # print("No stimulation detected")
            pulseCount = 0
            stimLength = 0
            return Params(0, 0, 0, 0, 0, 0, 0, 0)

        return Params(timestamp, delay, pulseCount, pulseDuration, delayBetweenPulses,
                      ampChangePerPulse, firstPulseAmp, stimLength)

    def stimLibrary(self):
        return self.params["StimulusLibrary"]

    def pipette(self):
        """Get pipette number

        Returns
        -------
        int
            Pipette number
        """
        return int(self.params["SessionIndex"].flatten()[0])

    def session(self):
        """Get session number

        Returns
        -------
        string
            Session number
        """
        sess = Path(self.inputFile)
        return sess.name.partition('.')[0]

    def date(self):
        """Get session date

        Returns
        -------
        string
            File date
        """
        date = self.inputFile.parent.stem
        return date

    def fileName(self):
        """Get complete file name

        Returns
        -------
        string
            File name
        """
        return self.inputFile

    def isStimEnabled(self):
        """Check whether stimulation was active in the given recording

        Returns
        -------
        int
            0 if stimulation was inactive, 1 if stimulation was active
        """
        return int(self.params["IsStimulationEnabled"])

    def sampleRate(self):
        """Get session sample rate. Probably 20kHz, but just to be sure.

        Returns
        -------
        float
            Sampling rate, in Hz
        """
        self.sample_rate = float(self.params["AcquisitionSampleRate"].flatten()[0])
        return self.sample_rate

    def downsampledRate(self):
        """Get downsampled sampling rate

        Returns
        -------
        int
            Downsampled rate
        """
        return int(self.sampleRate() / self.ds_factor)

    def clock(self):
        """Get clock/timestamp at start of recording, in POSIX time

        Returns
        -------
        float
            Time at start of recording. Useful for calculation of absolute elapsed time.
        """
        try:
            clock_at_start = [elem[0] for elem in self.params["ClockAtRunStart"]]

            YEAR = int(clock_at_start[0])
            MONTH = int(clock_at_start[1])
            DAY = int(clock_at_start[2])
            HOUR = int(clock_at_start[3])
            MIN = int(clock_at_start[4])
            SEC = round(clock_at_start[5], 14)  # we must round to evade a rare inifinite float

            clock_at_start = f"{YEAR}/{MONTH}/{DAY} {HOUR:02}{MIN:02}{SEC:02}"
            clock_at_start = (datetime.strptime(clock_at_start,
                                                "%Y/%m/%d %H%M%S.%f").astimezone().timestamp())

        except ValueError as msg:
            print(msg)
            clock_at_start = self.params["ClockAtRunStart"]

        return clock_at_start

    def units(self):
        """Get scale units for voltage and current

        Returns
        -------
        tuple of strings
            Tuple containing the voltage units first, and the current units second
        """
        volt_units = self.params["AIChannelUnits"][0].astype(str)
        curr_units = self.params["AIChannelUnits"][1].astype(str)
        time_units = 's'
        return volt_units, curr_units, time_units

    def channelScales(self):
        """Get scale factors for analog input and output channels

        Returns
        -------
        dict
            Scale factors for input and output channels in `V/V` (voltage AI and AO),
            `V/A` (current AI) and `A/V` (current AO)
        """
        mV = 1e-3
        pA = 1e-12
        scales = {
            'volt_ai': self.params["AIChannelScales"].flatten()[0] / mV,
            'volt_ao': self.params["AOChannelScales"].flatten()[1] * mV,
            'curr_ai': self.params["AIChannelScales"].flatten()[1] / pA,
            'curr_ao': self.params["AOChannelScales"].flatten()[0] * pA
        }

        return scales

    def getSweepDuration(self):
        """Get sweep duration if set to a finite number

        Returns
        -------
        float
            Sweep duration for given .h5 file
        """
        return float(self.params["SweepDurationIfFinite"])

    def areSweepsInf(self):
        """Checks if sweeps were acquired infinitely

        Returns
        -------
        int
            0 if sweeps are discrete, 1 if sweeps are continuous
        """
        return int(self.params["AreSweepsContinuous"])

    def runsCompleted(self):
        """Checks the number of runs completed

        Returns
        -------
        int
            Number of runs already completed, excluding the current run
        """
        return int(self.params["NRunsCompleted"])

    def nextRun(self):
        """Reads the absolute file name for the current run

        Returns
        -------
        str
            Absolute file path for current run
        """
        return self.params["NextRunAbsoluteFileName"].astype(str)

    def wsVersion(self):
        """Reads the WaveSurfer version used to acquire the data

        Returns
        -------
        str
            WaveSurfer version
        """
        return self.params["VersionString"].astype(str)

    def stimuli(self):
        """Read stimulus library

        Returns
        -------
        dict
            Dictionary with all stimulus metadata
        """
        return self.params["StimulusLibrary"]

    def isDownsampled(self):
        """Check whether the dataset was downsampled when accessed

        Returns
        -------
        bool
            Returns 0 if raw dataset was read, 1 if dataset was downsampled
        """
        return self.ds

    def downsampleFactor(self):
        """Read downsampling factor used when reading the dataset

        Returns
        -------
        int
            Downsampling factor
        """
        return self.ds_factor


if __name__ == '__main__':
    import doctest
    doctest.testmod(name='__init__')
