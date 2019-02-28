import filecmp
import pytest
import os
from pathlib import Path
from molehill.pipeline import Pipeline

TEST_DATA_DIR = Path(__file__).resolve().parent / 'resources'


@pytest.fixture(scope='function', autouse=True)
def change_dir(tmp_path):
    current_dir = os.curdir
    os.chdir(tmp_path)
    yield
    os.chdir(current_dir)


def test_dump_yaml():
    dig_file = Path("output.dig")
    input_yaml = TEST_DATA_DIR / "titanic_pipeline.yml"

    pipeline = Pipeline()
    pipeline.dump_pipeline(input_yaml, dig_file, False)
    assert dig_file.read_text() == (TEST_DATA_DIR / "titanic.dig").read_text()
    dc = filecmp.dircmp(TEST_DATA_DIR / 'queries', 'queries')
    for diff_file in dc.diff_files:
        assert (Path(TEST_DATA_DIR) / 'queries' / diff_file).read_text() == (Path('queries') / diff_file).read_text()

    assert len(dc.diff_files) == 0
