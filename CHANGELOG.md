# Changelog

## [Unreleased]

## [0.3.0] - 2020-08-24

### Added
- Support for multi-input models where alternate input does not merge with primary sequence input before a stop layer.
- Support for layers that dependend on exact order of inputs, e.g. Subtract and Concat.


## [0.2.0] - 2020-08-22

### Added
- Support for recursively defined networks with 3 test cases
- This Changelog file.

### Changed
- BPNet test cases atol changed to 1e-5 so they pass deterministically

## [0.1.3] - 2020-08-21
### Added
- First PyPI release and tagged version
- Tested and working on non-recursively defined single-input, single and multi-output architectures
- Tested and working on arcitectures with skip connections

--- 
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


[unreleased]: https://github.com/kundajelab/fastISM/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/kundajelab/fastISM/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/kundajelab/fastISM/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/kundajelab/fastISM/releases/tag/v0.1.3
