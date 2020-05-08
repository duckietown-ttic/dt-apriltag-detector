# Apriltag Detection Workspace

This repository contains a workspace for development of new features
in the apriltag detection pipeline.


## Requirements

Running this workspace requires the following tools and libraries:
- docker
- dts
- git


## Structure

This repository wraps the main apriltag repository containing the detection
pipeline (implemented in C) with a bunch of Python scripts.

The main pipeline is declared as a submodule of this repository and is linked
to the fork `https://github.com/ripl-ttic/apriltag3`.


## Clone the repository

Clone this repository using the following command:

```
git clone --recurse-submodules git@github.com:duckietown-ttic/dt-apriltag-detector
```

This will clone this repository and the `apriltag3` submodule repo.


## Build the Docker environment

Make sure that your DTS is set to use the `ente` version of the commands 
(i.e., `dts --set-version ente`).

You can build the Docker image containing the workspace environment by running 
the command 

```
dts devel build -a amd64 -f
```

**NOTE:** After you run this once, you can keep reusing the image even if your code
 changes (see next section for details), you will only need to re-run this when 
 either the Dockerfile or the dependencies change.


## Launch an experiment

Experiments are defined using DT Launchers. A Launcher is a bash script inside the
directory `launch`. This repo comes with some predefined launchers. If a launcher
is not specified, `default.sh` will be used.

Use the following command to run the default launcher:

```
dts devel run -a amd64 --mount
```

You can look at the launcher script `launch/default.sh` to see what this does.
A new directory will be created inside `packages/tests/`. Inside that directory, 
an HTML file called `viewer.html` contains the result of your tests, check it out.

Define your own launchers (feel free to copy an existing one and update it) and
launch it using the command

```
dts devel run -a amd64 --mount --launcher <YOUR_LAUNCHER_NAME>
```

For example, you can run

```
dts devel run --mount -a amd64 --launcher rectification_left
```