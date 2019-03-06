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
    test_query_dir = TEST_DATA_DIR / 'queries'
    test_dig_file = TEST_DATA_DIR / "titanic.dig"

    pipeline = Pipeline()
    pipeline.dump_pipeline(input_yaml, dig_file, False)
    assert dig_file.read_text() == test_dig_file.read_text()
    dc = filecmp.dircmp(test_query_dir, 'queries')
    for diff_file in dc.diff_files:
        assert (test_query_dir / diff_file).read_text() == (Path('queries') / diff_file).read_text()

    assert len(dc.diff_files) == 0


def test_dump_yaml_oversample():
    dig_file = Path("output.dig")
    input_yaml = TEST_DATA_DIR / "titanic_pipeline_oversample.yml"
    test_query_dir = TEST_DATA_DIR / 'queries_oversample'
    test_dig_file = TEST_DATA_DIR / "titanic_oversample.dig"

    pipeline = Pipeline()
    pipeline.dump_pipeline(input_yaml, dig_file, False)
    assert dig_file.read_text() == test_dig_file.read_text()
    dc = filecmp.dircmp(test_query_dir, 'queries')
    for diff_file in dc.diff_files:
        assert (test_query_dir / diff_file).read_text() \
               == (Path('queries') / diff_file).read_text()

    assert len(dc.diff_files) == 0


def test_dump_yaml_pos_oversample():
    dig_file = Path("output.dig")
    input_yaml = TEST_DATA_DIR / "titanic_pipeline_pos_oversample.yml"
    test_query_dir = TEST_DATA_DIR / 'queries_pos_oversample'
    test_dig_file = TEST_DATA_DIR / "titanic_pos_oversample.dig"

    pipeline = Pipeline()
    pipeline.dump_pipeline(input_yaml, dig_file, False)
    assert dig_file.read_text() == test_dig_file.read_text()
    dc = filecmp.dircmp(test_query_dir, 'queries')
    for diff_file in dc.diff_files:
        assert (test_query_dir / diff_file).read_text() \
               == (Path('queries') / diff_file).read_text()

    assert len(dc.diff_files) == 0


def test_dump_yaml_randomforest():
    dig_file = Path("output.dig")
    input_yaml = TEST_DATA_DIR / "titanic_pipeline_rf.yml"
    test_query_dir = TEST_DATA_DIR / 'queries_rf'
    test_dig_file = TEST_DATA_DIR / "titanic_rf.dig"

    pipeline = Pipeline()
    pipeline.dump_pipeline(input_yaml, dig_file, False)
    assert dig_file.read_text() == test_dig_file.read_text()
    dc = filecmp.dircmp(test_query_dir, 'queries')
    for diff_file in dc.diff_files:
        assert (test_query_dir / diff_file).read_text() \
               == (Path('queries') / diff_file).read_text()

    assert len(dc.diff_files) == 0
