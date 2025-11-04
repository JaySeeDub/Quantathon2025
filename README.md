# SCQuantum-SRNL-Challenge-2025
Repository for the SRNL Challenge at SCQuantum QuantathonV2. 

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/TariniHardikar/SCQuantum-SRNL-Challenge-2025.git)

```markdown
# TornadoQ


### Install
```bash
pip install -e .[dev]
```


### CLI
```bash
python -m tornadoq.cli clean-data --help
python -m tornadoq.cli clean-data --train TRAIN.xlsx --test TEST.xlsx
python -m tornadoq.cli benchmark-binary --train TRAIN.xlsx --test TEST.xlsx --folds 5 --repeats 2
python -m tornadoq.cli benchmark-multiclass --train TRAIN.xlsx --test TEST.xlsx
```
