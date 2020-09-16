# Changelog

## [Unreleased]

## [0.4.0] - 2020-09-16

### Added
- Sequences for benchmarking in notebooks dir and a notebook to process the sequence
- Benchmarking notebooks
- Notebook to time Basset conv and fc separately 
- Ability to specify custom mutations
- For each mutation, models only run on input sequences for which character is different from mutation. As a result, each batch usually has a different size. This is slow for the first few batches as it entails a one-time cost.  
- Lots of documentation and a logo!

### Changed
- Models updated:
  - Activation added to Basset
  - Num output for Basset and Factorized Basset
  - For BPNet, only one channel output and one counts instead of two

### Fixed
- FastISM object would keep intermediate outputs of a batch even after it was used, as a result it would occupy extra memory. Get rid of such objects now through a `cleanup()` function. This has stopped GPU Resource errors that popped up after running a few batches

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


[unreleased]: https://github.com/kundajelab/fastISM/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/kundajelab/fastISM/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/kundajelab/fastISM/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/kundajelab/fastISM/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/kundajelab/fastISM/releases/tag/v0.1.3
