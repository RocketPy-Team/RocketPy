# RocketPy Change Log

All notable changes to `RocketPy` project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

<!-- Types of changes:
    - `Added` for new features.
    - `Changed` for changes in existing functionality.
    - `Deprecated` for soon-to-be removed features.
    - `Removed` for now removed features.
    - `Fixed` for any bug fixes.
    - `Security` in case of vulnerabilities.

    Should not be here:
    - Tests and test updates.
    - GitHub maintenance tasks (repository reorganization or CI changes, etc.)
    - Merge commits, as they usually don't contain information valuable to the end-user.
    - Small refactors that don't impact the functionality or improve performance.
    - Minor changes such as "updated a notebook", "updated readme", or other
      documentation tweaks unless they significantly enhance understanding or usability.
    - In summary: if your change doesn't impact other codes or doesn't offer a
    significant improvement for the final user, it probably shouldn't be here.

    Types of messages:
    - The primary message is usually the title of the PR and its number.
    - If the PR encompasses a wide range of changes (which ideally it shouldn't),
      you can use a second line or a brief list to describe these changes succinctly.
    - Ensure the description is clear and understandable even for those who
      might not have deep technical knowledge of the project.
-->

## [Unreleased] - yyyy-mm-dd

Here we write upgrading notes for brands. It's a team effort to make them as
straightforward as possible.

### Added

-

### Changed

-

### Fixed

-

## [v1.1.4] - 2023-12-07

You can install this version by running `pip install rocketpy==1.1.4`

### Fixed

- FIX: changes Generic Motor exhaust velocity to cached property [#497](https://github.com/RocketPy-Team/RocketPy/pull/497)

## [v1.1.3] - 2023-11-29

You can install this version by running `pip install rocketpy==1.1.3`

### Fixed

- FIX: Broken Function.get_value_opt for N-Dimensional Functions [#492](https://github.com/RocketPy-Team/RocketPy/pull/492)

## [v1.1.2] - 2023-11-27

You can install this version by running `pip install rocketpy==1.1.2`

### Fixed

- BUG: Function breaks if a header is present in the csv file [#485](https://github.com/RocketPy-Team/RocketPy/pull/485)

## [v1.1.1] - 2023-11-24

You can install this version by running `pip install rocketpy==1.1.1`

### Added

- DOC: Added this changelog file [#472](https://github.com/RocketPy-Team/RocketPy/pull/472)
- ENH: Prevent out of bounds Tanks from Instantiation #484 [#484](https://github.com/RocketPy-Team/RocketPy/pull/484)

### Fixed

- HOTFIX: Negative Static Margin [#476](https://github.com/RocketPy-Team/RocketPy/pull/476)
- HOTFIX: 2D .CSV Function and missing set_get_value_opt call [#478](https://github.com/RocketPy-Team/RocketPy/pull/478)
- HOTFIX: Tanks Overfill not Being Detected [#479](https://github.com/RocketPy-Team/RocketPy/pull/479)

## [v1.1.0] - 2023-11-19

You can install this version by running `pip install rocketpy==1.1.0`

### Added

- DOC: Documentation for Function Class Usage [#465](https://github.com/RocketPy-Team/RocketPy/pull/465)
- DOC: first simulation all_info [#466](https://github.com/RocketPy-Team/RocketPy/pull/466)
- ENH: draw motors [#436](https://github.com/RocketPy-Team/RocketPy/pull/436)
- DOC: add documentation for flight data export. [#464](https://github.com/RocketPy-Team/RocketPy/pull/464)
- ENH: Add mass_flow_rate() to GenericMotor class [#459](https://github.com/RocketPy-Team/RocketPy/pull/459)
- DOC: Add documentation on how to build the docs [#452](https://github.com/RocketPy-Team/RocketPy/pull/452)
- ENH: new Flight.get_solution_at_time() method [#441](https://github.com/RocketPy-Team/RocketPy/pull/441)
- ENH: rocket drawing [419](https://github.com/RocketPy-Team/RocketPy/pull/419)
- ENH: Adding Stability Margin with Mach dependency [#377](https://github.com/RocketPy-Team/RocketPy/pull/377)

### Changed

- ENH: Spherical Caps Included in Total Length [#455](https://github.com/RocketPy-Team/RocketPy/pull/455)
  - Important: This changes behavior of `TankGeometry.add_spherical_caps()`
- ENH: Clean Plots and Prints sub packages init files [#457](https://github.com/RocketPy-Team/RocketPy/pull/457)
- ENH: Add \_MotorPlots Inheritance to Motor Plots Classes [#456](https://github.com/RocketPy-Team/RocketPy/pull/456)
- DOC: organize flight examples folder [#429](https://github.com/RocketPy-Team/RocketPy/pull/429)
- DOC: improve mass and inertia docs [#445](https://github.com/RocketPy-Team/RocketPy/pull/445)

### Fixed

- MNT: Refactor exhaust velocity calculation to avoid ZeroDivisionError [#470](https://github.com/RocketPy-Team/RocketPy/pull/470)
- BUG: Fix find_input() Function to Return a Single Value [#471](https://github.com/RocketPy-Team/RocketPy/pull/471)
- DOC: refactor dispersion analysis notebook [#463](https://github.com/RocketPy-Team/RocketPy/pull/463)
- BUG: User input checks added for Function class [#451](https://github.com/RocketPy-Team/RocketPy/pull/451)
- DOC: fix positions and coordinate system documentation page [#454](https://github.com/RocketPy-Team/RocketPy/pull/)
- MNT: fix env plots legends [#440](https://github.com/RocketPy-Team/RocketPy/pull/440)
- BUG: flight.prints.max_values() fails when launching an EmptyMotor [#438](https://github.com/RocketPy-Team/RocketPy/pull/438)
- BUG: Maintaining Extrapolation when Adding Discrete Functions with Constants [#432](https://github.com/RocketPy-Team/RocketPy/pull/432)
- MNT: Fix env plots max heights [#433](https://github.com/RocketPy-Team/RocketPy/pull/433)
