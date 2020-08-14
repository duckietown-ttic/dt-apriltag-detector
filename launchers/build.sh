#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable CODE_DIR to know the absolute path to your code
# NOTE: Use `dt_exec COMMAND` to run the main process (blocking process)

# launching app
dt_exec catkin build --workspace /code/catkin_ws/


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
