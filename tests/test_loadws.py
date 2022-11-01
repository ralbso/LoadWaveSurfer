from loadwavesurfer.loadws import LoadWaveSurfer as loadws
import os
from pathlib import Path

def test_file_load():
    curr_directory = Path(os.path.realpath(__file__)).parent
    filename = os.path.join(curr_directory, 'p8_0005.h5')

    f = loadws(filename)

    assert isinstance(f.data(), tuple)
    assert f.sampleRate() == 20e3
