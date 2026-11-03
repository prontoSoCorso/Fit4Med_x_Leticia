"""Reading the example .mvnx gives the expected table and recording information."""
import pandas as pd
import pytest

import config
from pipeline.mvnx import mvnx_to_csv, recording_info


@pytest.mark.skipif(not config.EXAMPLE_MVNX.exists(), reason="example .mvnx not available")
def test_read_example_mvnx(tmp_path):
    header, n_frames = mvnx_to_csv(config.EXAMPLE_MVNX, tmp_path / "recording.csv")
    table = pd.read_csv(tmp_path / "recording.csv")
    info = recording_info(header, config.EXAMPLE_MVNX)
    assert n_frames == len(table) == 1538 and table.shape[1] == 1 + 788
    assert (info["test"], info["timepoint"], info["take"]) == ("10MWT", "t0", 4)
    assert "jointAngle_jRightKnee_z" in table and table.notna().all().all()
