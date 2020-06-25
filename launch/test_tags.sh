#!/bin/bash

source /environment.sh

# initialize launch file
dt_launchfile_init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

CATKIN_WS="/code/catkin_ws"
APRILTAG_PACKAGE_DIR="${CATKIN_WS}/src/dt-apriltag-detector"
DATA_DIR="${APRILTAG_PACKAGE_DIR}/dataset_generator"
TESTS_DIR="${APRILTAG_PACKAGE_DIR}/dataset_generator/tests"
VIEWER_TEMPLATE="double"
VIEWER_PATH="${APRILTAG_PACKAGE_DIR}/packages/viewer_templates/${VIEWER_TEMPLATE}.html"

if [ -n "$EXPERIMENT" ]; then
  EXPERIMENTS=("_${EXPERIMENT}")
else
  EXPERIMENTS=("raw" "rect")
fi


for EXP in "${EXPERIMENTS[@]}"; do

  # constants
  DATA_TYPE="${EXP}"

  # test configuration
  IN_FOLDER="${DATA_DIR}/${DATA_TYPE}/*"
  i=0
  for INPUT in $IN_FOLDER; do

    TEST_IMAGE="${INPUT}"
    IN_FILE="${INPUT}"
    SAMPLENR="sample_$i"

    OUT_DIR="${TESTS_DIR}/${SAMPLENR}/${EXP}"
    # make out dir
    mkdir -p "${OUT_DIR}"

    # launching app
    (
      cd ${OUT_DIR} || exit;
      ${CATKIN_WS}/devel/bin/apriltag_demo \
          --debug \
          --decimate=1.0 \
          --refine-edges=0 \
          ${IN_FILE};
          mogrify -format jpg ./*.pnm;
          rm -f ./*.pnm
    )
    i=i+1

  done

  # create viewer
  VIEWER_OUT="${OUT_DIR}/../viewer.html"
  cp ${VIEWER_PATH} "${VIEWER_OUT}"
  sed -i "s/__EXP1__/raw/g" "${VIEWER_OUT}"
  sed -i "s/__EXP2__/rect/g" "${VIEWER_OUT}"
  sed -i "s/__EXPERIMENT__/${DATA_TYPE}/g" "${VIEWER_OUT}"

done

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# terminate launch file
dt_launchfile_terminate
